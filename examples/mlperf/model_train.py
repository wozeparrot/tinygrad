import functools
import os
import time
from tqdm import tqdm

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv, BEAM, WINO
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save

from examples.mlperf.helpers import get_training_state, load_training_state

def train_resnet():
  from extra.models import resnet
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup
  from examples.mlperf.initializers import Conv2dHeNormal, Linear
  from examples.hlb_cifar10 import UnsyncedBatchNorm

  config = {}
  seed = config["seed"] = getenv("SEED", 42)
  Tensor.manual_seed(seed)  # seed for weight initialization

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]

  # ** model definition and initializers **
  num_classes = 1000
  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = resnet.ResNet50(num_classes)

  # shard weights and initialize in order
  for k, x in get_state_dict(model).items():
    if not getenv("SYNCBN") and ("running_mean" in k or "running_var" in k):
      x.realize().shard_(GPUS, axis=0)
    else:
      x.realize().to_(GPUS)
  parameters = get_parameters(model)

  # ** hyperparameters **
  epochs            = config["epochs"]            = getenv("EPOCHS", 41)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 8.5 * (BS/2048))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 5)
  decay             = config["decay"]             = getenv("DECAY", 2e-4)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_epochs       = getenv("EVAL_EPOCHS", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (len(get_train_files()) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (len(get_val_files()) // EVAL_BS)

  config["BEAM"]    = BEAM.value
  config["WINO"]    = WINO.value
  config["SYNCBN"]  = getenv("SYNCBN")

  # ** Optimizer **
  from examples.mlperf.optimizers import LARS
  skip_list = {v for k, v in get_state_dict(model).items() if "bn" in k or "bias" in k or "downsample.1" in k}
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay, skip_list=skip_list)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, initial_lr=base_lr, end_lr=1e-4,
                                        train_steps=epochs * steps_in_train_epoch,
                                        warmup=lr_warmup_epochs * steps_in_train_epoch)
  print(f"training with batch size {BS} for {epochs} epochs")

  # ** resume from checkpointing **
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer, scheduler, safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming from {ckpt} at epoch {start_epoch}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args)

  BENCHMARK = getenv("BENCHMARK")

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return x.permute([0, 3, 1, 2]) - input_mean
  @TinyJit
  def train_step(X, Y):
    optimizer.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = out.sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.realize(), top_1.realize()
  @TinyJit
  def eval_step(X, Y):
    X = normalize(X)
    out = model.forward(X)
    loss = out.sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    return loss.realize(), top_1.realize()
  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), cookie

  # ** epoch loop **
  step_times = []
  for e in range(start_epoch, epochs):
    # ** train loop **
    Tensor.training = True
    it = iter(tqdm(batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e),
                   total=steps_in_train_epoch, desc=f"epoch {e}", disable=BENCHMARK))
    i, proc = 0, data_get(it)
    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      (loss, top_1_acc), proc = train_step(proc[0], proc[1]), proc[2]

      pt = time.perf_counter()

      try:
        next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss, top_1_acc = loss.numpy().item(), top_1_acc.numpy().item() / BS

      cl = time.perf_counter()
      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/top_1_acc": top_1_acc, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": e + (i + 1) / steps_in_train_epoch})

      st = cl
      proc, next_proc = next_proc, None  # return old cookie
      i += 1

      if i == BENCHMARK:
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_hours = median_step_time * steps_in_train_epoch * epochs / 60 / 60
        print(f"Estimated training time: {estimated_total_hours:.0f}h{(estimated_total_hours - int(estimated_total_hours)) * 60:.0f}m")
        return

    # ** eval loop **
    if (e + 1 - eval_start_epoch) % eval_epochs == 0:
      train_step.reset()  # free the train step memory :(
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      Tensor.training = False

      it = iter(tqdm(batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False), total=steps_in_val_epoch))
      proc = data_get(it)
      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        (loss, top_1_acc), proc = eval_step(proc[0], proc[1]), proc[2]  # drop inputs, keep cookie

        try:
          next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        loss, top_1_acc = loss.numpy().item(), top_1_acc.numpy().item() / EVAL_BS
        eval_loss.append(loss)
        eval_top_1_acc.append(top_1_acc)
        proc, next_proc = next_proc, None  # return old cookie

        et = time.time()
        eval_times.append(et - st)

      eval_step.reset()
      total_loss = sum(eval_loss) / len(eval_loss)
      total_top_1 = sum(eval_top_1_acc) / len(eval_top_1_acc)
      total_fw_time = sum(eval_times) / len(eval_times)
      tqdm.write(f"eval loss: {total_loss:.2f}, eval time: {total_fw_time:.2f}, eval top 1 acc: {total_top_1:.3f}")
      if WANDB:
        wandb.log({"eval/loss": total_loss, "eval/top_1_acc": total_top_1, "eval/forward_time": total_fw_time, "epoch": e + 1})

      # save model if achieved target
      if not achieved and total_top_1 >= target:
        fn = f"./ckpts/resnet50.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
        achieved = True

      # checkpoint every time we eval
      if getenv("CKPT"):
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        safe_save(get_training_state(model, optimizer, scheduler), fn)

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  from extra.models.bert import BertForPreTraining
  from extra.datasets.wikipedia import iterate, get_train_files
  from extra.lr_scheduler import CosineAnnealingLR

  mdl = BertForPreTraining()
  mdl.load_from_pretrained()

  from tinygrad.nn.optim import LAMB
  optimizer = LAMB(get_parameters(mdl), lr=1e-4, wd=0.01)
  lr_scheduler = CosineAnnealingLR(optimizer, len(get_train_files()), eta_min=1e-6)

  if getenv("WANDB"):
    import wandb
    wandb.init(project="tinygrad-mlperf-bert")

  @TinyJit
  def eval_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids):
    pred, _ = mdl(input_ids, input_mask, segment_ids)
    acc = mdl.accuracy(pred, masked_lm_positions, masked_lm_ids)
    return acc.realize()

  @TinyJit
  def train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels):
    pred, seq = mdl(input_ids, input_mask, segment_ids)
    loss = mdl.loss(pred, seq, masked_lm_positions, masked_lm_ids, next_sentence_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.realize()

  print(f"there are {len(get_train_files())} training files")

  done = False
  for i in range(300):
    # train loop
    for j, (X, _) in enumerate(iterate(bs=4)):
      input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
      masked_lm_positions, masked_lm_ids, next_sentence_labels = Tensor(X["masked_lm_positions"], requires_grad=False), Tensor(X["masked_lm_ids"], requires_grad=False), Tensor(X["next_sentence_labels"], requires_grad=False)
      # split across ranks
      input_ids, input_mask, segment_ids = input_ids.chunk(world_size, 0)[rank], input_mask.chunk(world_size, 0)[rank], segment_ids.chunk(world_size, 0)[rank]
      masked_lm_positions, masked_lm_ids, next_sentence_labels = masked_lm_positions.chunk(world_size, 0)[rank], masked_lm_ids.chunk(world_size, 0)[rank], next_sentence_labels.chunk(world_size, 0)[rank]
      loss = train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels)
      lr_scheduler.step()

      # update wandb
      if getenv("WANDB"):
        wandb.log({
          "loss": loss.numpy().item(),
          "lr": optimizer.lr.numpy().item(),
        })

      # eval loop
      if j % 10000 == 0:
        # save checkpoint
        safe_save(get_state_dict(mdl), f"weights/ckpt_{i}_{j}.bert.safetensors")

        train_step.reset()
        with Tensor.train(False):
          accuracies = []
          for X, _ in iterate(bs=8, val=True):
            input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
            masked_lm_positions, masked_lm_ids = Tensor(X["masked_lm_positions"], requires_grad=False), Tensor(X["masked_lm_ids"], requires_grad=False)
            input_ids, input_mask, segment_ids = input_ids.chunk(world_size, 0)[rank], input_mask.chunk(world_size, 0)[rank], segment_ids.chunk(world_size, 0)[rank]
            masked_lm_positions, masked_lm_ids = masked_lm_positions.chunk(world_size, 0)[rank], masked_lm_ids.chunk(world_size, 0)[rank]
            acc = eval_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids).item()
            accuracies.append(acc)

          final_acc = accuracies / len(accuracies)
          if getenv("WANDB"): wandb.log({"eval_accuracy": final_acc})
          if final_acc >= 0.72:
            print(f"achieved 0.72 accuracy, stopping")
            done = True
            break
        eval_step.reset()
    if done: break

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()
