import os
import sys

sys.path.append("../")
import argparse
from tqdm.auto import tqdm
from PIL import Image

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from transformers import LlamaForCausalLM, LlamaTokenizer

from modules.lora import monkeypatch_or_replace_lora_extended
from modules.adapters import TextAdapter
from huggingface_hub import hf_hub_download
from argparse import ArgumentParser

# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--llama2_dir", type=str, default="")
    parser.add_argument("--prompts_path", type=str, default="inputs/in.txt")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--generate_text", action="store_true")
    parser.add_argument("--dont_use_lora", action="store_true")
    parser.add_argument("--is_gradio", action="store_true")

    args = parser.parse_args()
    return args


def main(args, prompts, is_gradio=False):
    print("Starting the main function")
    os.makedirs(args.output_dir, exist_ok=True)
    prompts_path = args.prompts_path
    if os.path.exists(prompts_path):
        with open(prompts_path, "r") as f:
            prompts = f.readlines()
    else:
        prompts = prompts


    # download the model from the Hugging Face model hub # meta-llama/Llama-2-7b-hf

    # meta-llama/Llama-2-7b-hf
    # print("Downloading the model from the Hugging Face model hub")
    # hf_hub_download("meta-llama/Llama-2-7b-hf", './/') #, args.llama2_dir)
    # args.llama2_dir = "meta-llama/Llama-2-7b-hf"
    # print("Downloaded the model from the Hugging Face model hub")

    VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
    TEXT_ENCODER_REPLACE_MODULES = {"LlamaAttention"}
    height = 512
    width = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    torch_device = "cuda"
    dont_use_lora = args.dont_use_lora
    pos_prompt = ", best quality, extremely detailed, 4k resolution"
    lora_ending = '_no_lora' if dont_use_lora else ''

    use_chat = args.use_chat
    generate_text = args.generate_text
    input_ids_max_length = 77
    if use_chat:
        args.llama2_dir = "meta-llama/Llama-2-7b-chat-hf"
    chat_flag = "_chat" if use_chat else ""

    # Modules of T2I diffusion models
    print("Loading the modules of T2I diffusion models")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device)
    vis = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    print(f"Loading a tokenizer from {args.llama2_dir}")
    tokenizer = LlamaTokenizer.from_pretrained(args.llama2_dir)
    # To perform inference on a 24GB GPU memory, llama2 was converted to half precision
    print(f"Loading a text encoder from {args.llama2_dir}")
    text_encoder = LlamaForCausalLM.from_pretrained(args.llama2_dir, torch_dtype=torch.float16).to(torch_device)
    print(f"Loading an adapter from {os.path.join(args.ckpt_dir, f'adapter')}")
    adapter = TextAdapter.from_pretrained(os.path.join(args.ckpt_dir, f"adapter")).to(torch_device)
    print("Loaded all the modules of T2I diffusion models")
    tokenizer.pad_token = '[PAD]'

    # LoRA - visual encoder
    monkeypatch_or_replace_lora_extended(
        vis,
        torch.load(os.path.join(args.ckpt_dir, f"lora_vis.pt")),
        r=32,
        target_replace_module=VIS_REPLACE_MODULES,
    )

    if dont_use_lora:
        pass
    else:
        # LoRA - text encoder
        monkeypatch_or_replace_lora_extended(
            text_encoder,
            torch.load(os.path.join(args.ckpt_dir, f"lora_text.pt")),
            r=32,
            target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
        )


    vae.eval()
    vis.eval()
    text_encoder.eval()
    adapter.eval()

    images = []
    # Inference
    with torch.no_grad():
        for prompt in prompts:
            print(prompt)
            orig_prompt = prompt
            # no need for pos_prompt for diffusion lens
            # prompt += pos_prompt

            # Text embeddings
            text_ids = tokenizer(prompt,
                                 padding="max_length",
                                 max_length=input_ids_max_length,
                                 return_tensors="pt",
                                 truncation=True).input_ids.to(torch_device)


            if generate_text:
                from transformers import GenerationConfig
                generation_config = GenerationConfig(output_hidden_states=True)
                text_ids_for_generation = tokenizer(prompt,
                                            max_length=input_ids_max_length,
                                            return_tensors="pt",
                                            truncation=True).input_ids.to(torch_device)

                text_embeddings_generated = text_encoder.generate(input_ids=text_ids_for_generation,
                                                        max_length=77,
                                                        num_return_sequences=1,
                                                        no_repeat_ngram_size=2,
                                                        early_stopping=True,
                                                        generation_config=generation_config,
                                                        return_dict_in_generate=True)

                generate_ids = text_embeddings_generated['sequences']

                file = open(f"{args.output_dir}/generated_text.txt", "w")
                file.write(tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0])
                file.close()

                print(f"Generated text: {tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]}")

            text_embeddings = text_encoder(input_ids=text_ids,
                                           output_hidden_states=True,
                                           return_dict=True) # .hidden_states[-1].to(torch.float32)




            # print(f"Text embeddings: {text_embeddings}")
            # print(f"Text embeddings shape: {text_embeddings.shape}")

            hidden_states = text_embeddings.hidden_states

            # if hasattr(text_encoder.base_model, 'final_layer_norm'):
            #     print("Yes! text_encoder has attribute final_layer_norm")
            #     final_layer_norm = text_encoder.base_model.final_layer_norm
            # else:
            #     print("No! text_encoder does not have attribute final_layer_norm")
            #     print(dir(text_encoder.base_model))

            for hidden_stat_index, hidden_state in enumerate(hidden_states):
                # print("Len: ", len(hidden_states))
                # print(f"Hidden state[0] {hidden_stat_index} shape: {hidden_state[0].shape}")
                # print(f"Hidden state[1] {hidden_stat_index} shape: {hidden_state[1].shape}")
                # hidden_state = final_layer_norm(hidden_state)
                # hidden_state = hidden_state[-1]

                hidden_state = hidden_state.to(torch.float32)

                hidden_state = adapter(hidden_state).sample

                print(f"Text embeddings shape after adapter.sample: {hidden_state.shape}")
                uncond_input = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
                # Convert the text embedding back to full precision
                uncond_embeddings = \
                text_encoder(uncond_input.input_ids.to(torch_device), output_hidden_states=True).hidden_states[-1].to(
                    torch.float32)
                uncond_embeddings = adapter(uncond_embeddings).sample
                text_embeddings = torch.cat([uncond_embeddings, hidden_state])

                # Latent preparation
                latents = torch.randn((1, vis.in_channels, height // 8, width // 8)).to(torch_device)
                latents = latents * noise_scheduler.init_noise_sigma

                # Model prediction
                noise_scheduler.set_timesteps(num_inference_steps)
                for t in tqdm(noise_scheduler.timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    noise_pred = vis(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                # Decoding
                latents = 1 / 0.18215 * latents
                image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                image = Image.fromarray(image)
                prompt_dir = os.path.join(args.output_dir, f'{orig_prompt[: 30]}{lora_ending}{chat_flag}')
                if not os.path.exists(prompt_dir):
                    os.makedirs(prompt_dir)

                image.save(f'{prompt_dir}/{hidden_stat_index:02d}.png')
                print(f"Saved the image to path: {prompt_dir}/{hidden_stat_index:02d}.png")
                if is_gradio:
                    images.append(image)
    if is_gradio:
        return images


if __name__ == "__main__":
    print("Starting the main function")


    prompts = [
        'What is the color of the sky?',
        'How many hearts does an octopus have?',
        'What is the capital of France?',
        'What is the capital of Germany?',
        'Who is the president of the United States?',
    ]

    args = parse_args()
    main(args, prompts)

    # 'A sunny day, after the rain.',
    # 'A woman propose marriage to a man.',
    # 'A man taking a shot',
    # 'A man taking a shot in the forest.',
    # 'A man taking a shot in a bar.',
    # 'A female doctor examining a patient.',
    # 'A blue cat and a red dog playing soccer.',

    # "قِرَان",
    # "婚禮",
    # "शादी",

    # "A pear sink in water.",
    # "A sunnier day, after the rain.",
    # "A hot pan with corn seeds after a few minutes.",
    # "A hot pan with corn seeds after a few seconds.",
    # "A hot pan with corn seeds after a few hours.",

    # "Yes or no: Would a pear sink in water?",
    # "Why Albert Einstein can't hold a smartphone?",
    # "What happens to a pot of boiling water when it is left on the stove for too long?",
    # "свадьба",
    # "What happens when a person is exposed to extreme cold?",
    # "Which population is bigger California or Texas?",
    # "Who is the president of the United States?",
    # "How many fingers does a human have?",
    # "How many arms does the octopus have?",
    # "How many hearts does the octopus have?",
    # "дом",
    # "What happens when an egg falls on the floor?",

    # prompts = [
    #     "city",
    #     "Oppenheimer sits on the beach on a chair, watching a nuclear exposition with a huge mushroom cloud, 120mm.",
    #     "Tiny potato kings wearing majestic crowns, sitting on thrones, overseeing their vast potato kingdom filled with potato subjects and potato castles.",
    #     "An illustration of a human heart made of translucent glass, standing on a pedestal amidst a stormy sea. Rays of sunlight pierce the clouds, illuminating the heart, revealing a tiny universe within.",
    #     "freshly made hot floral tea in glass kettle on the table.",
    #     "In homage to old-world botanical sketches, an illustration is rendered with detailed lines and subtle watercolor touches. The artwork captures an unusual fusion: a cactus bearing not just thorns but also the fragrant and delicate blooms of a lilac, all while taking on the mesmerizing form of a Möbius strip, capturing the essence of nature’s diverse beauty and mathematical intrigue.",
    #     "Portrait photography, a woman in a glamorous makeup, wearing a mask with tassels, in the style of midsommar by Ari Aster, made of flowers, bright pastel colors, prime lense.",
    #     "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    #     "Dark high contrast render of a psychedelic tree of life illuminating dust in a mystical cave.",
    # ]
