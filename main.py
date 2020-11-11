# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader
from trainer import trainer
from datasets import dataset
from LiarLiar import arePantsonFire
from Encoder import Encoder
from utils import infer
from Attention import MultiHeadAttention, PositionFeedforward


# Do not change module_list , otherwise no marks will be awarded
liar_dataset_train=dataset(purpose='test_class')
dataloader_train=DataLoader(dataset=liar_dataset_train, batch_size=1)
print('train data loaded')
liar_dataset_val=dataset(prep_Data_from='val', purpose='test_class')
dataloader_val=DataLoader(dataset=liar_dataset_val, batch_size=1)
print('data loaded')

statement_encoder=Encoder(conv_layers=5, hidden_dim=512)
justification_encoder=Encoder(conv_layers=5, hidden_dim=512)

multiheadAttention=MultiHeadAttention(hid_dim=512, n_heads=32)
positionFeedForward=PositionFeedforward(hid_dim=512,feedForward_dim=2048)

print('creating and training model')
model=arePantsonFire(sentence_encoder=statement_encoder, explanation_encoder=justification_encoder,multihead_Attention=multiheadAttention,position_Feedforward=positionFeedForward, hidden_dim=512,max_length_sentence=liar_dataset_train.get_max_lenghts()[0],max_length_justification=liar_dataset_train.get_max_lenghts()[1],input_dim=200)
trainer(model,dataloader_train,dataloader_val,1)
print('done training')
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test', purpose='test_class')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)
