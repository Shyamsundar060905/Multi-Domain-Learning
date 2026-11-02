#!/usr/bin/env bash
# Individual (single-dataset) runs of the reconstruction + trainable-encoder
# U-Net -- LEVIR, WHU, SYSU -- back to back on one GPU, with a summary file.
#
#   ./run_indiv_recon.sh                                              # config defaults: 40 epochs, step 15, recon 0.5
#   GPU=cuda:6 RECON_WEIGHT=0.1 EPOCHS=150 STEP=50 ./run_indiv_recon.sh   # match the 150-epoch joint run
#
#   # or one dataset per GPU, in parallel (each in its own screen session):
#   GPU=cuda:1 DATASETS=levir RECON_WEIGHT=0.1 EPOCHS=150 STEP=50 ./run_indiv_recon.sh
#
# Use the SAME recon weight, epochs and step as the joint (3-dataset) run, or the
# individual and joint numbers are not comparable.  Each run's log, checkpoint
# folder and summary file are named with these settings, so nothing is overwritten.

cd "$(dirname "$0")" || exit 1
GPU=${GPU:-cuda:0}
RECON_WEIGHT=${RECON_WEIGHT:-0.5}
EPOCHS=${EPOCHS:-40}
STEP=${STEP:-15}
DATASETS=${DATASETS:-levir whu sysu}
mkdir -p logs
SUMMARY=logs/indiv_recon_summary_w${RECON_WEIGHT}_${EPOCHS}ep_s${STEP}_$(echo $DATASETS | tr ' ' '-').txt

echo "=== started $(date) on $(git log --oneline -1) | recon_weight=$RECON_WEIGHT | epochs=$EPOCHS step=$STEP | $GPU ===" >> "$SUMMARY"
for D in $DATASETS; do
  if [ ! -f "configs/indiv_unet_recon_trainable_$D.json" ]; then
    echo "unknown dataset '$D' (use levir, whu, sysu)" | tee -a "$SUMMARY"
    continue
  fi
  NAME=indiv_unet_recon_trainable_${D}_w${RECON_WEIGHT}_${EPOCHS}ep_s${STEP}
  echo "--- $D  started $(date)" >> "$SUMMARY"
  python -u main.py --config "configs/indiv_unet_recon_trainable_$D.json" \
      --recon-weight "$RECON_WEIGHT" --epochs "$EPOCHS" --scheduler-step-size "$STEP" \
      --ckpt-dir "checkpoints/$NAME" --device "$GPU" \
      > "logs/$NAME.log" 2>&1
  echo "exit code: $?   finished $(date)" >> "$SUMMARY"
  tr '\r' '\n' < "logs/$NAME.log" \
    | grep -E "loaded:|\[cap\]|Reconstruction loss|Trainable parameters|Restored best checkpoint|test\] main aggregate|test\] recon|Average Dice" \
    >> "$SUMMARY"
  echo >> "$SUMMARY"
done
echo "=== finished $(date) ===" >> "$SUMMARY"
