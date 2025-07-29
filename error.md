Traceback (most recent call last):
  File "main.py", line 63, in <module>
    main()
  File "main.py", line 51, in main
    train(config, writer, train_dataloader)
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/train.py", line 197, in train
    ret = [criterion(f'train/{n}', out) for n, out in zip(names, decode_image)]
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/train.py", line 197, in <listcomp>
    ret = [criterion(f'train/{n}', out) for n, out in zip(names, decode_image)]
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/loss.py", line 104, in forward
    output_mask_feature_arr = clip.encode_img(self.clip_model, prep_arr[1], prep_arr[2])
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/AlphaCLIP/clip.py", line 63, in encode_img
    image_features = model.visual(image, alpha)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/AlphaCLIP/alpha_clip/model.py", line 365, in forward
    x = self.transformer(x, return_attn=False)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/AlphaCLIP/alpha_clip/model.py", line 331, in forward
    return self.resblocks(x)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/AlphaCLIP/alpha_clip/model.py", line 275, in forward
    attn_out, attn = self.attention(self.ln_1(x))
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/AlphaCLIP/alpha_clip/model.py", line 272, in attention
    return self.attn(x, attn_mask=self.attn_mask)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/seohyun/miniconda3/envs/ver2/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/seohyun/Desktop/seohyun/70. Project/ver2.0/AlphaCLIP/alpha_clip/model.py", line 243, in forward
    attn = attn.softmax(dim=-1)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 MiB. GPU 
(ver2) seohyun@seohyun-Precision-5820-Tower:~/Desktop/seohyun/70. Project/ver2.0$ 
