py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py --phase teacher --data_dir "C:\Users\KOOL-GUY69\Desktop\multieye_data\assemble_oct" --save_dir ".\checkpoints\teacher" --img_size 224