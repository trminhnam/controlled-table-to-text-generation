python trainer.py --mode train \
        --model T5 \
        --num_epochs 2 \
        --cuda \
        --batch_size 8 \
        --train_input '/home/sreehs/dlt/table-to-text-generation/Data/totto_data_small/processed_totto_train_data_1000.jsonl' \
        --development_input '/home/sreehs/dlt/table-to-text-generation/Data/totto_data_small/processed_totto_dev_data_500.jsonl'
