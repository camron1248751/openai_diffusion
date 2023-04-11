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

def create_argparser():
    defaults = dict(
        similarity_scale=100,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



dist_util.setup_dist()
logger.configure()

logger.log("creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.load_state_dict(
    dist_util.load_state_dict(args.model_path, map_location="cpu")
)
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()

input_image_path = "imgs/seven.png"
input_image = Image.open(input_image_path).convert("RGB")
input_image = input_image.resize((args.image_size, args.image_size))
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
classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
classifier.load_state_dict(
    dist_util.load_state_dict(args.classifier_path, map_location="cpu")
)
classifier.to(dist_util.dev())
if args.classifier_use_fp16:
    classifier.convert_to_fp16()
classifier.eval()
input_class_label = get_input_image_class(input_image, classifier, dist_util.dev())

cond_fn = create_cond_fn(input_image, similarity_scale=args.similarity_scale)


def model_fn(x, t, y=None):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            l2_distance = F.mse_loss(x_in, input_image.expand_as(x_in))
            grad = th.autograd.grad(l2_distance, x_in)[0] * args.similarity_scale
            x = x_in - grad

        batch_size = x.size(0)
        class_labels = th.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
        )
        batch_class_labels = input_class_label.repeat(batch_size).to(dist_util.dev())
        batch_class_labels = th.where(class_labels == input_class_label, batch_class_labels, class_labels)
        return model(x, t, batch_class_labels)

def gen(args):
    # for multiple images
    # input_image_paths = ["path/to/image1.png", "path/to/image2.png", "path/to/image3.png"]

    # def model_fn(x, t, y=None):
    #     batch_class_labels = input_class_label.repeat(x.size(0)).to(dist_util.dev())
    #     return model(x, t, batch_class_labels)

        #return model(x, t, y if args.class_cond else None)
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        batch_class_labels = input_class_label.repeat(args.batch_size).to(dist_util.dev())
        model_kwargs["y"] = batch_class_labels
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        print('i')
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
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
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join("/Users/camronsallade/Documents/gon/ML/cs238/final/guided-diffusion", f"samples_{shape_str}.npz") #logger.get_dir()
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")



if __name__ == "__main__":
    gen(args)
