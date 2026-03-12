1. git clone
   ```bash
   git clone git@github.com:saifahmadgit/go2-sim2real-locomotion-rl.git
   ```

2. Create virtual env
   ```bash
   cd ~/go2-sim2real-locomotion-rl
   python3 -m venv venv
   source venv/bin/activate
   ```

3. pip install -e .
   ```bash
   pip install -e .
   ```

4. pip install rsl-rl-lib==2.2.4
   ```bash
   pip install rsl-rl-lib==2.2.4
   ```

5. pip install pynput (for running eval scripts, like walking it can be used to teleop)
   ```bash
   pip install pynput
   ```

6. pip install tensorboard (for monitoring the progress of training)
   ```bash
   pip install tensorboard
   ```

7. keep the logs, where you are running your scripts from, the script searches for logs at the same level, for both evaluation and training

8. Training ,  for stairs training I started from the base walk trained check point, stair training script has --resume to start from a check pont
   env for jump and crouch are the base env provided by Genesis and some modifications, In process of updating scripts to achieve sim to real transfer
   the walk and stair env are custom, Following commands can be used to train the policies

   Walk:
   ```bash
   python3 examples/locomotion/final/go2_train_walk.py -e test1 --max_iterations 100
   ```

   Stair:
   ```bash
   python3 examples/locomotion/final/go2_train_stair.py -e test1 --max_iterations 100 --resume logs/go2-walk/model_188000.pt
   ```

   Crouch:
   ```bash
   python3 examples/locomotion/final/go2_train_crouch.py -e test1 --max_iterations 100
   ```

   Jump:
   ```bash
   python3 examples/locomotion/final/go2_train_crouch.py -e test1 --max_iterations 100
   ```

9. Evaluation

   P - forward
   M - backward
   J/K - side movement
   U/O - Yaw

   The checkpoins tat are working is included in the repo which can be directly run using following:

   Walk:
   ```bash
   python3 examples/locomotion/final/go2_eval_walk.py -e go2-walk --ckpt 188000
   ```

   Stairs:
   ```bash
   python3 examples/locomotion/final/go2_eval_stairs.py -e go2-stairs --ckpt 104000
   ```

   Crouch:
   ```bash
   python3 examples/locomotion/final/go2_eval_base.py -e go2-crouch --ckpt 2999
   ```

   Jump:
   ```bash
   python3 examples/locomotion/final/go2_eval_base.py -e go2-jump --ckpt 999
   ```
