"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import json
import cv2

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)




def guided_gen(input_image):
    args = {
        "attention_resolutions": "32,16,8",
        "batch_size": 4,
        "channel_mult": "",
        "class_cond": True,
        "classifier_attention_resolutions": "32,16,8",
        "classifier_depth": 4,
        "classifier_path": "../models/64x64_classifier.pt",
        "classifier_pool": "attention",
        "classifier_resblock_updown": True,
        "classifier_scale": 1.0,
        "classifier_use_fp16": False,
        "classifier_use_scale_shift_norm": True,
        "classifier_width": 128,
        "clip_denoised": True,
        "diffusion_steps": 10000,
        "dropout": 0.1,
        "image_size": 64,
        "learn_sigma": True,
        "model_path": "../models/64x64_diffusion.pt",
        "noise_schedule": "cosine",
        "num_channels": 192,
        "num_head_channels": 64,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "num_res_blocks": 3,
        "num_samples": 4,
        "output_folder": "uploads",
        "predict_xstart": False,
        "resblock_updown": True,
        "rescale_learned_sigmas": False,
        "rescale_timesteps": False,
        "similarity_scale": 3000,
        "timestep_respacing": "5",
        "use_checkpoint": False,
        "use_ddim": False,
        "use_fp16": False,
        "use_kl": False,
        "use_new_attention_order": True,
        "use_scale_shift_norm": True,
    }
    model_args = {
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "timestep_respacing": "5",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": True,
        "image_size": 64,
        "class_cond": True,
        "learn_sigma": True,
        "num_channels": 192,
        "num_res_blocks": 3,
        "channel_mult": "",
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "attention_resolutions": "32,16,8",
        "dropout": 0.1
    }

    classifier_args = {
        "image_size": 64,
        "classifier_use_fp16": False,
        "classifier_width": 128,
        "classifier_depth": 4,
        "classifier_attention_resolutions": "32,16,8",
        "classifier_use_scale_shift_norm": True,
        "classifier_resblock_updown": True,
        "classifier_pool": "attention"
    }
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **model_args
    )
    model.load_state_dict(
        dist_util.load_state_dict(args["model_path"], map_location="cpu")
    )
    model.to(dist_util.dev())
    if args["use_fp16"]:
        model.convert_to_fp16()
    model.eval()

    input_image = input_image.resize((args["image_size"], args["image_size"]))
    input_image = th.from_numpy(np.array(input_image)).permute(2, 0, 1).float()
    input_image = (input_image / 127.5) - 1.0  # Normalize to the range [-1, 1]\


    def create_cond_fn(input_image, similarity_scale=1.0):
            input_image = input_image.to(dist_util.dev())
            input_image = input_image.unsqueeze(0)

            def cond_fn(x, t, y=None):
                with th.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    l2_distance = F.mse_loss(x_in, input_image.expand_as(x_in))
                    grad = th.autograd.grad(l2_distance, x_in)[0] * similarity_scale
                    return grad

            return cond_fn

    def get_input_image_class(input_image, classifier, device):
            input_image = input_image.unsqueeze(0).to(device)
            logits = classifier(input_image, th.tensor([1.0]).to(device))
            probs = F.softmax(logits, dim=-1)
            _, class_label = th.max(probs, dim=-1)
            return class_label

    logger.log("loading classifier...")
    classifier = create_classifier(**classifier_args)
    classifier.load_state_dict(
        dist_util.load_state_dict(args["classifier_path"], map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args["classifier_use_fp16"]:
        classifier.convert_to_fp16()
    classifier.eval()
    input_class_label = get_input_image_class(input_image, classifier, dist_util.dev())

    cond_fn = create_cond_fn(input_image, similarity_scale=args["similarity_scale"])


    def model_fn(x, t, y=None):
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                l2_distance = F.mse_loss(x_in, input_image.expand_as(x_in))
                grad = th.autograd.grad(l2_distance, x_in)[0] * args["similarity_scale"]
                x = x_in - grad

            batch_size = x.size(0)
            class_labels = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
            )
            batch_class_labels = input_class_label.repeat(batch_size).to(dist_util.dev())
            batch_class_labels = th.where(class_labels == input_class_label, batch_class_labels, class_labels)
            return model(x, t, batch_class_labels)


    def transfer_color(src, dst):
        src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)

        for channel in range(3):
            src_hist, src_bins = np.histogram(src[:, :, channel].ravel(), 256, [0, 256])
            dst_hist, dst_bins = np.histogram(dst[:, :, channel].ravel(), 256, [0, 256])

            src_cdf = src_hist.cumsum()
            dst_cdf = dst_hist.cumsum()

            src_cdf_normalized = src_cdf * (dst_cdf[-1] / src_cdf[-1])

            transfer_function = np.interp(src_cdf_normalized, dst_cdf, dst_bins[:-1])

            src[:, :, channel] = cv2.LUT(src[:, :, channel], transfer_function.astype('uint8'))

        return cv2.cvtColor(src, cv2.COLOR_LAB2RGB)


    def gen(args):
        # for multiple images
        # input_image_paths = ["path/to/image1.png", "path/to/image2.png", "path/to/image3.png"]

        # def model_fn(x, t, y=None):
        #     batch_class_labels = input_class_label.repeat(x.size(0)).to(dist_util.dev())
        #     return model(x, t, batch_class_labels)

            #return model(x, t, y if args.class_cond else None)
        exp = int(args["timestep_respacing"])
        logger.log(f"sampling... expected time {exp * 5} seconds...")
        all_images = []
        all_labels = []
        while len(all_images) * args["batch_size"] < args["num_samples"]:
            model_kwargs = {}
            batch_class_labels = input_class_label.repeat(args["batch_size"]).to(dist_util.dev())
            model_kwargs["y"] = batch_class_labels
            sample_fn = (
                diffusion.p_sample_loop if not args["use_ddim"] else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model_fn,
                (args["batch_size"], 3, args["image_size"], args["image_size"]),
                clip_denoised=args["clip_denoised"],
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            gathered_labels = [th.zeros_like(input_class_label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, input_class_label)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            n_samples = len(all_images) * args["batch_size"]
            logger.log(f"created {n_samples} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args["num_samples"]]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args["num_samples"]]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(args['output_folder'], f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr, label_arr)

            generated_filenames = []
            # Save the generated image(s) as PNG
            for idx, img_arr in enumerate(arr):
                img = Image.fromarray(img_arr)
                generated_filename = f"generated_{idx}.png"
                img.save(os.path.join(args['output_folder'], generated_filename))
                generated_filenames.append(generated_filename)
            # for idx, img_arr in enumerate(arr):
            #     print(img_arr.shape)
            #     img = Image.fromarray(img_arr)
            #     input_image_np = np.array(input_image)
            #     print(input_image_np.shape)
            #     img_np = np.array(img)
            #     img_np = transfer_color(img_np, input_image_np)
            #     img = Image.fromarray(img_np)
            #     generated_filename = f"generated_{idx}.png"
            #     img.save(os.path.join(args['output_folder'], generated_filename))
            #     generated_filenames.append(generated_filename)

            # Return the list of generated image filenames
            dist.barrier()
            logger.log("sampling complete")
            print(json.dumps(generated_filenames))
            return generated_filenames
    files = gen(args)
    print(json.dumps(files))
    #return files

import argparse

# Add the following at the beginning of the script
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image.')
    return parser.parse_args()

if __name__ == "__main__":
    #args = parse_args()
    input_image_path = "uploads/apple.png"
    input_image = Image.open(input_image_path).convert("RGB")
    guided_gen(input_image)
