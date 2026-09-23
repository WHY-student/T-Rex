<div align="center">

We modified our model based on T-Rex: Tactile-Reactive Dexterous Manipulation

</div>

## Repository layout

```text
T-Rex/
├── qwen_vla/                       # three-expert MoT model + VLA wrapper
│   ├── modeling_qwen3vl_mot.py     # Qwen3VLAttentionMoT, decoder layer, MoT model
│   ├── modeling_vla.py             # Qwen3VLVLAModel: ViT + MoT + embedders
│   │                               # forward_flow_action_{full,partial},
│   │                               # tactile_flow_continue, tactile_flow_train_step
│   ├── diffusion.py                # ActionEmbedder, TimestepEmbedder, FinalLayer
│   ├── DeformAE.py                 # DeformEncoder for tactile-deformation images
│   └── lerobot_dataset.py          # LeRobot v3.0 dataloader (TRexLeRobotDataset)
├── tactile_vqvae/                  # tactile VQ-VAE model (used by the embedded tokenizer)
├── scripts/                        # post-train + ZMQ inference server
│   ├── train.sh + train.py         # post-train SFT (fine-tune from a midtrain ckpt)
│   ├── test.sh + test.py            # ZMQ inference server
│   └── lora_test.py + lora_test.sh # strict Action-LoRA inference launcher
├── utils/                          # data prep + checkpoint tooling
│   ├── gen_json_tac_deltabase_eef_bimanual_parallel.py + gen_json_bimanual.sh
│   │                               # raw task data → training JSON (eef-62)
│   ├── convert_inlab_to_lerobot.py (+ .sh)
│   │                               # raw task data → LeRobot v3.0 (eef-62)
│   ├── lerobot_common.py           # shared schema + pose math + norm stats
│   ├── encode_vqvae_codes_to_json.py (+ .sh)
│   │                               # optional code pre-baker
│   ├── merge_vqvae_into_ckpt.py (+ .sh)
│   │                               # bake VQ-VAE into a checkpoint
│   └── analyze_episode.py          # per-episode visualization
├── config/sft_qwen.yaml            # accelerate + DeepSpeed config
├── dataset_quickstart/             # standalone companion: browse / inspect / replay the dataset
├── hardware_code/                  # teleoperation + data-collection stack (robot hardware code)
└── pyproject.toml                  # pinned dependencies
```

## Citation

If you find T-Rex useful, please cite:

```bibtex
@misc{trex2026,
  title={T-Rex: Tactile-Reactive Dexterous Manipulation}, 
  author={Dantong Niu and Zhuoyang Liu and Zekai Wang and Boning Shao and Zhao-Heng Yin and Anirudh Pai and Yuvan Sharma and Stefano Saravalle and Ruijie Zheng and Jing Wang and Ryan Punamiya and Mengda Xu and Yuqi Xie and Yunfan Jiang and Letian Fu and Konstantinos Kallidromitis and Matteo Gioia and Junyi Zhang and Jiaxin Ge and Haiwen Feng and Fabio Galasso and Wei Zhan and David M. Chan and Yutong Bai and Roei Herzig and Jiahui Lei and Fei-Fei Li and Ken Goldberg and Jitendra Malik and Pieter Abbeel and Yuke Zhu and Danfei Xu and Jim Fan and Trevor Darrell},
  year={2026},
  eprint={2606.17055},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.17055}, 
}
```
<!-- TODO: replace the title / author / arXiv id above with the real values. -->
