CUDA_VISIBLE_DEVICES=0 MUJOCO_GL="glx" xvfb-run -s "-screen 0 1280x720x24" -a python ../experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint "/tmp/Distill-VLA/runs/configs+libero_10_no_noops+b1+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-ActionQueryAlign--libero_10_no_noops--20260209_184127--20000_chkpt" \
    --task_suite_name "libero_10" \
    --use_pro_version True \
    > "eval_logs/Long--chkpt.log" 2>&1 &