import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def trainer(model, train_dataloader, val_dataloader, num_epochs, path_to_save='/home/atharva',
          checkpoint_path='/home/atharva',
          checkpoint=100, train_batch=1, test_batch=1, device='cuda:0'): # 2 Marks. 
      """
      Everything by default gets shifted to the GPU. Select the device according to your system configuration
      If you do no have a GPU, change the device parameter to "device='cpu'"
      :param model: the Classification model..
      :param train_dataloader: train_dataloader
      :param val_dataloader: val_Dataloader
      :param num_epochs: num_epochs
      :param path_to_save: path to save model
      :param checkpoint_path: checkpointing path
      :param checkpoint: when to checkpoint
      :param train_batch: 1
      :param test_batch: 1
      :param device: Defaults on GPU, pass 'cpu' as parameter to run on CPU. 
      :return: None
      """
      torch.backends.cudnn.benchmark = True 
      model.train()
      model.cuda()
      criterion=nn.CrossEntropyLoss().cuda()
      optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

      max_acc=None
      training_loss=[]
      val_loss=[]
      training_acc=[]
      val_acc=[]

      for epoch in range(1,num_epochs+1):
            if (epoch%checkpoint) == 0:
                  torch.save({
                      'epoch':epoch,
                      'optimizer':optimizer.state_dict(),
                      'model':model.state_dict(),
                      'train_loss':training_loss,
                      'val_loss':val_loss,
                      'train_acc':training_acc,
                      'val_acc':val_acc
                  }, checkpoint_path)
                  exit(10)
            epoch_train_loss=0
            epoch_val_loss=0
            epoch_acc_train=0
            epoch_val_accuracy=0
            model.train()
            for _, data in enumerate(train_dataloader):
                  print('starting train cycle')
                  optimizer.zero_grad()
                  data['statement']=data['statement'].cuda()
                  data['justification']=data['justification'].cuda()
                  data['credit_history']=data['credit_history'].cuda()
                  #data['label']=torch.squeeze(data['label'])
                  label=data['label'][0]
                  label=label.cuda()
                  #data['label']=data['label'].cuda()
                  #print(data['label'][0])

                  output=model(data['statement'],data['justification'],data['credit_history']).unsqueeze(0)
                  #print(output)
                  loss=criterion(output, label.long())
                  print('starting backprop')
                  loss.backward()
                  optimizer.step()
                  epoch_train_loss+=loss.item()
                  _ ,predicted=torch.max(output.data,1)
                  epoch_acc_train+=(predicted==label).sum().item()
                  del data['statement'],data['justification'],data['label'],data['credit_history'],label
            training_loss.append(epoch_train_loss/(_*train_batch))
            training_acc.append(epoch_acc_train/_*train_batch)

            with torch.no_grad():
                  model.eval()
                  epoch_val_loss=0
                  epoch_val_accuracy=0
                  for _, data in enumerate(val_dataloader):
                        optimizer.zero_grad()
                        data['statement']=data['statement'].cuda()
                        data['justification']=data['justification'].cuda()
                        data['credit_history']=data['credit_history'].cuda()
                        label=data['label'][0]
                        label=label.cuda()
                        #data['label']=data['label'].cuda()

                        output=model(data['statement'],data['justification'],data['credit_history']).unsqueeze(0)
                        loss=criterion(output, label)
                        epoch_val_loss+=loss.item()
                        _ ,predicted=torch.max(output.data,1)
                        epoch_val_accuracy+=(predicted==label).sum().item()
                        del data['statement'],data['justification'],data['label'],data['credit_history'], label
                  val_loss.append(epoch_val_loss/(_*test_batch))
                  val_acc.append(epoch_val_accuracy/_*test_batch)

                  if max_acc is None:
                        max_acc=epoch_val_accuracy/(_*test_batch)
                  else:
                        if (epoch_val_accuracy/(_*test_batch)) > max_acc:
                              max_acc=epoch_val_accuracy/(_*test_batch)
                              torch.save(model.state_dict(),path_to_save)
            print(epoch_acc_train)
            print(epoch_train_loss)
            print(epoch_val_accuracy)
            print(epoch_val_loss)

      plt.plot(training_acc)
      plt.plot(val_acc)
      plt.plot(training_loss)
      plt.plot(val_loss)
      plt.show()
      return None

