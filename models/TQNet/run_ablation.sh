#!/bin/bash

# Ablation studies.
# Modify the boolean values of parameters `self.use_tq` and `self.channel_aggre` in `models/TQNet` to perform ablation studies
#1. When `self.use_tq=True` and `self.channel_aggre=True`, it represents the original TQNet.
#2. When `self.use_tq=True` and `self.channel_aggre=False`, it indicates the removal of the attention module, but the TQ design is retained, and this part becomes the channel identifier module.
#3. When `self.use_tq=False` and `self.channel_aggre=True`, it indicates the removal of TQ, but the attention module is retained, and this part becomes the self-attention module.
#4. When `self.use_tq=False` and `self.channel_aggre=False`, it represents the removal of both TQ and the attention module, leaving only a basic MLP module.
sh scripts/Ablation/TQNet.sh;

# Integration studies
sh scripts/Ablation/DLinear.sh;
sh scripts/Ablation/iTransformer.sh;
sh scripts/Ablation/PatchTST.sh;
sh scripts/Ablation/TQDLinear.sh;
sh scripts/Ablation/TQiTransformer.sh;
sh scripts/Ablation/TQPatchTST.sh;
