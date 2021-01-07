# Controlled Table-To-Text Generation

Please refer to our project report for details on the work and our results: [Project Report](https://github.com/mon95/controlled-table-to-text-generation/blob/master/Project_Final_Report_Controlled_ToTTo.pdf)

Note: This repo has been moved (partially) from our internal private repository to serve as documentation. For more info, please reach out over email.



Use following command to run the training script.
```
python run.py --mode train \
                --model Bert2Bert \
                --num_epochs 2\
                --cuda \
                --batch_size 8\
                --train_input '/content/drive/My Drive/DLT/table-to-text-generation/Data/totto_data_small/totto_train_data_1000.jsonl'\
                --development_input '/content/drive/My Drive/DLT/table-to-text-generation/Data/totto_data_small/totto_dev_data_500.jsonl'
               
```

To run the T5 model, simply run:

```
bash train.sh
```
This invokes the trainer.py script with T5 as the model and the processed data files (will need to change the data file locations accordingly)

Code:

Preprocessing and Evaluation Scripts are borrowed from the [ToTTo repository](https://github.com/google-research/language/tree/master/language/totto)


References:

```
@article{parikh2020totto,
  title={ToTTo: A Controlled Table-To-Text Generation Dataset},
  author={Parikh, Ankur P and Wang, Xuezhi and Gehrmann, Sebastian and Faruqui, Manaal and Dhingra, Bhuwan and Yang, Diyi and Das, Dipanjan},
  journal={arXiv preprint arXiv:2004.14373},
  year={2020}
```

