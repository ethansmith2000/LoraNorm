pip install peft -y
pip uninstall peft -y
python ./train_lora.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --instance_prompt "sks toy" --validation_prompt "a sks toy on the moon" --center_crop --gradient_checkpointing --allow_tf32 --mixed_precision "bf16" --enable_xformers_memory_efficient_attention --rank 16 --instance_data_dir ./dataset/ --max_train_steps 1200