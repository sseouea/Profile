# Profile

hello

(base) seohyun@seohyun-Precision-5820-Tower:~/Desktop/seohyun$ lspci | grep -i nvidia
17:00.0 VGA compatible controller: NVIDIA Corporation TU106 [GeForce RTX 2070] (rev a1)
17:00.1 Audio device: NVIDIA Corporation TU106 High Definition Audio Controller (rev a1)
17:00.2 USB controller: NVIDIA Corporation TU106 USB 3.1 Host Controller (rev a1)
17:00.3 Serial bus controller: NVIDIA Corporation TU106 USB Type-C UCSI Controller (rev a1)
65:00.0 VGA compatible controller: NVIDIA Corporation GP106GL [Quadro P2000] (rev a1)
65:00.1 Audio device: NVIDIA Corporation GP106 High Definition Audio Controller (rev a1)

device check: cuda cuda:0
device check: cuda cuda:1
Couldn't connect to the Hub: 429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/runwayml/stable-diffusion-v1-5.
Will try to load from local cache.
Couldn't connect to the Hub: 429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/runwayml/stable-diffusion-v1-5.
Will try to load from local cache.
[rank1]: Traceback (most recent call last):
[rank1]:   File "main.py", line 61, in <module>
[rank1]:     main()
[rank1]:   File "main.py", line 50, in main
[rank1]:     train(config, writer, train_dataloader)
[rank1]:   File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/train.py", line 81, in train
[rank1]:     unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/accelerator.py", line 1350, in prepare
[rank1]:     result = tuple(
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/accelerator.py", line 1351, in <genexpr>
[rank1]:     self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/accelerator.py", line 1226, in _prepare_one
[rank1]:     return self.prepare_model(obj, device_placement=device_placement)
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/accelerator.py", line 1477, in prepare_model
[rank1]:     model = torch.nn.parallel.DistributedDataParallel(
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 800, in __init__
[rank1]:     _sync_module_states(
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/distributed/utils.py", line 298, in _sync_module_states
[rank1]:     _sync_params_and_buffers(process_group, module_states, broadcast_bucket_size, src)
[rank1]:   File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/distributed/utils.py", line 309, in _sync_params_and_buffers
[rank1]:     dist._broadcast_coalesced(
[rank1]: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 274.00 MiB. GPU  has a total capacity of 4.93 GiB of which 137.69 MiB is free. Including non-PyTorch memory, this process has 4.30 GiB memory in use. Of the allocated memory 4.14 GiB is allocated by PyTorch, and 35.88 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[Epoch 1]:   0%|                                                             | 0/1 [00:00<?, ?it/s]W0729 12:42:14.712448 125974634772288 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 20798 closing signal SIGTERM
E0729 12:42:15.153488 125974634772288 torch/distributed/elastic/multiprocessing/api.py:826] failed (exitcode: 1) local_rank: 1 (pid: 20799) of binary: /home/seohyun/miniconda3/envs/ver2/bin/python
Traceback (most recent call last):
  File "/home/seohyun/miniconda3/envs/ver2/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/commands/accelerate_cli.py", line 48, in main
    args.func(args)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/commands/launch.py", line 1159, in launch_command
    multi_gpu_launcher(args)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/accelerate/commands/launch.py", line 793, in multi_gpu_launcher
    distrib_run.run(args)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/distributed/run.py", line 870, in run
    elastic_launch(
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 263, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
main.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-07-29_12:42:14
  host      : seohyun-Precision-5820-Tower
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 20799)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
