import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoTokenizer, GPT2Config
from kv_causal_attention import KVGPT2Model
import time



def build_config():
    # increase vocab size to a multiple of 16, e.g. 50304 for performance
    # Learned the trick from Karpathy, good stuff!
    config_args = dict(
        n_layer=12, n_head=12, n_embd=768, vocb_size=50304, block_size=1024, resid_pdrop=0.1
    )
    config = GPT2Config(**config_args)
    return config


def build_model():
    # Load the Hugging Face GPT-2 pretrained model
    gpt2_model = GPT2LMHeadModel.from_pretrained(
        "gpt2", cache_dir="/home/fin2rng/.cache/huggingface/hub"
    )
    print(gpt2_model)

    # Load your custom PagedAttention GPT-2 model (Karpathy-like)
    config = build_config()
    kvgpt2_model = KVGPT2Model(config)  # Assuming PagedAttention is in place
    #print(kvgpt2_model)

    # kvgpt2_model.load_state_dict(gpt2_model.state_dict(), strict=False)

    # Compare the state dict keys between the Hugging Face model and custom model
    gpt2_state_dict = gpt2_model.state_dict()
    kvgpt2_state_dict = kvgpt2_model.state_dict()

    # check_weight_match(gpt2_state_dict, kvgpt2_state_dict)
    copy_model_weight(
        config, gpt2_model, kvgpt2_model, gpt2_state_dict, kvgpt2_state_dict
    )

    return gpt2_model, kvgpt2_model


def check_weight_match(gpt2_state_dict: dict, kvgpt2_state_dict: dict) -> bool:
    matched = True
    for key, custom_key in zip(gpt2_state_dict.keys(), kvgpt2_state_dict.keys()):
        hf_tensor = gpt2_state_dict[key]
        custom_tensor = kvgpt2_state_dict[custom_key]

        if hf_tensor.shape != custom_tensor.shape:
            print(
                f"Shape mismatch: {key} -> {hf_tensor.shape} vs {custom_key} -> {custom_tensor.shape}"
            )
            matched = False
        else:
            print(f"Matched: {key} -> {custom_key}")

    return matched


def copy_model_weight(
    config: GPT2Config,
    gpt2_model: GPT2LMHeadModel,
    kvgpt2_model: KVGPT2Model,
    gpt2_state_dict: dict,
    kvgpt2_state_dict: dict,
):
    """Copy the weights from transformer GPT to our KV GPT model.

    We need to transpose Conv1D weights to match Linear weights by transposing its dimensions
    Conv1D weight shape: (output dimension, input dimension) = (3*768, 768)
    Linear weight shape: (input features, kqv combined output features) = (768, 3*768)
    """
    with torch.no_grad():
        # Optionally load weights step-by-step and print any errors
        for key in gpt2_state_dict.keys():
            try:
                if key in kvgpt2_state_dict and "attn" not in key:
                    # if key in kvgpt2_state_dict:
                    kvgpt2_state_dict[key].copy_(gpt2_state_dict[key])
                else:
                    #print(f"Skipping {key}")
                    pass
            except Exception as e:
                print(f"Error loading {key}: {e}")

        for i in range(config.n_head):
            try:
                kvgpt2_model.transformer.h[i].attn.c_attn.weight.copy_(
                    gpt2_model.transformer.h[i].attn.c_attn.weight.t()
                )
                kvgpt2_model.transformer.h[i].attn.c_attn.bias.copy_(
                    gpt2_model.transformer.h[i].attn.c_attn.bias.t()
                )
                kvgpt2_model.transformer.h[i].attn.c_proj.weight.copy_(
                    gpt2_model.transformer.h[i].attn.c_proj.weight.t()
                )
                kvgpt2_model.transformer.h[i].attn.c_proj.bias.copy_(
                    gpt2_model.transformer.h[i].attn.c_proj.bias.t()
                )

            except Exception as e:
                print(
                    f"Error replacing attention weight transformer.h{i}.attn..weight: {e}"
                )


def benchmark(gpt2_model, kvgpt2_model, n_trials=10):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    kvgpt2_model.to(device)
    gpt2_model.to(device)
    prompt = "In the year 2045, artificial intelligence had reached a point where it was integrated into every aspect " \
             "of daily life. Machines were no longer tools, but companions, helping people with everything from decision-making " \
            "to creative endeavors. Governments and corporations around the world were rushing to establish ethical guidelines, " \
            "as AI systems became more autonomous, blurring the lines between human and machine intelligence. Autonomous vehicles " \
            "filled the streets, while drones patrolled the skies, delivering packages and assisting in law enforcement. Virtual "\
            "assistants had become so advanced that it was difficult to distinguish them from actual humans in conversation. "\
            "The future looked promising, but with great power came great responsibility. Scientists and engineers were focused "\
            "on ensuring that AI development would not lead to unintended consequences or social inequalities."\
            "As society became increasingly dependent on AI, a new challenge emerged: the question of sentience. Could these "\
            "machines, designed to learn and evolve, develop emotions or a sense of self? Some believed that AI consciousness "\
            "was inevitable, a natural evolution of their learning algorithms. Others, however, were more skeptical, warning of "\
            "the dangers of creating a system that might one day surpass human control. Public debates raged over whether AI should "\
            "have rights and whether safeguards were enough to protect humanity from an intelligence that might one day no longer require "\
            "human oversight."\
            "Meanwhile, AI continued to transform industries. In medicine, AI-assisted surgeries and diagnostics saved countless lives. "\
            "In education, AI tutors adapted lessons to fit individual students' needs, bridging gaps in traditional learning. The arts were "\
            "not immune to AI’s reach either, as algorithms capable of creating music, paintings, and even literature sparked both admiration "\
            "and controversy. Could a machine truly understand human creativity, or was it simply mimicking patterns learned from historical data? "\
            "Despite the rapid advancements, not all was perfect. Job displacement became a growing concern as automation replaced many traditional "\
            "roles. Governments scrambled to implement universal basic income, and new jobs emerged in fields like AI oversight and machine ethics, "\
            "but the transition was not smooth. Protests erupted in major cities, where workers demanded protections and policies to address the "\
            "widening gap between those who benefited from the AI revolution and those who were left behind. "\
            "In this new world, a delicate balance had to be maintained. While AI had the potential to solve many of humanity’s greatest challenges,"\
            "from climate change to disease eradication, it also introduced new risks. The world watched closely, knowing that every breakthrough came "\
            "with both the promise of progress and the shadow of unforeseen consequences."
    
    prompt = "The future of AI is"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    print(f"Number of tokens: {input_ids.size(1)}")
    attention_mask = inputs["attention_mask"]

    # Use a better sampling strategy for generation
    generation_kwargs = {
        "use_cache": False,
        "max_new_tokens": 50,  # Generate up to 1024 tokens
        "do_sample": True,  # Enable sampling
        "top_p": 0.95,  # Use nucleus sampling
        "temperature": 0.7,  # Lower temperature for more diverse outputs
        "repetition_penalty": 1.0,  # Penalty for repeated tokens
        "attention_mask": attention_mask,  # Pass the attention mask to avoid attending to padding
        "pad_token_id": tokenizer.eos_token_id,  # Set the pad token ID
    }

    generation_kwargs_kv = {
        "use_cache": True,
        "max_new_tokens": 50,  # Generate up to 1024 tokens
        "do_sample": True,  # Enable sampling
        "top_p": 0.95,  # Use nucleus sampling
        "temperature": 0.7,  # Lower temperature for more diverse outputs
        "repetition_penalty": 1.0,  # Penalty for repeated tokens
        "attention_mask": attention_mask,  # Pass the attention mask to avoid attending to padding
        "pad_token_id": tokenizer.eos_token_id,  # Set the pad token ID
    }

    with torch.no_grad():
        # start = time.time()
        # outputs = kvgpt2_model(input_ids, attention_mask=attention_mask, use_cache=True)
        # logits = outputs.logits

        # next_token_logits = logits[:, -1, :]
        # top_k = 10
        # probs = torch.softmax(next_token_logits, dim=-1)
        # top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

        # print(f"Top {top_k} predicted tokens and their probabilities:")
        # for i in range(top_k):
        #     token = tokenizer.decode(top_k_indices[0, i])
        #     prob = top_k_probs[0, i].item()
        #     print(f"Token: {token}, Probability: {prob}")

        # # Optionally, decode the most probable next token
        # next_token = tokenizer.decode(torch.argmax(next_token_logits, dim=-1))
        # print(f"Predicted next token: {next_token}")

        
        # kvgpt2_output = kvgpt2_model.generate(input_ids, **generation_kwargs)
        # # print("kvgp2 generate time: ", time.time() - start)
        # kvgpt2_text = tokenizer.decode(kvgpt2_output[0], skip_special_tokens=True)
        # print(f"KV GPT-2 Output: {kvgpt2_text}")
        
        start = time.time()
        gpt2_output = gpt2_model.generate(input_ids, **generation_kwargs)
        print("gp2 generate time w/o KV-Cache: ", time.time() - start)
        gpt2_text = tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
        print(
            "------------------------------------------------------------------------------------------------"
        )
        print(f"GPT-2 Output: {gpt2_text}")

        start = time.time()
        gpt2_output = gpt2_model.generate(input_ids, **generation_kwargs_kv)
        print("gp2 generate time w/ KV-Cache: ", time.time() - start)
        gpt2_text = tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
        print(
            "------------------------------------------------------------------------------------------------"
        )
        print(f"GPT-2 Output: {gpt2_text}")




if __name__ == "__main__":
    torch.cuda.empty_cache()
    gpt2_model, kv2gpt_model = build_model()
    benchmark(gpt2_model, kv2gpt_model)
