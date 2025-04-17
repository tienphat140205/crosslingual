import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.DualTopic import DualTopic
import os
import json
from torch.utils.data import DataLoader

class Runner:
    def __init__(self, args, params_list, lang1, lang2):
        self.args = args
        self.lang1 = lang1
        self.lang2 = lang2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualTopic(*params_list).to(self.device)
        self.log_file_path = None

    def make_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def make_lr_scheduler(self, optimizer):
        if getattr(self.args, 'lr_scheduler', None) != 'StepLR':
            return None
        if not hasattr(self.args, 'lr_step_size') or not hasattr(self.args, 'lr_gamma'):
            print("Warning: lr_scheduler is StepLR but lr_step_size or lr_gamma is missing in config. Scheduler disabled.")
            return None
        return StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma, verbose=False)

    def train(self, data_loader):
        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()
        lr_scheduler = self.make_lr_scheduler(optimizer)

        all_epoch_losses = []
        self.model.weight_MI = getattr(self.args, 'weight_MI', 1.0) 
        save_frequency = getattr(self.args, 'save_every_epochs', self.args.epochs + 1)

        print(f"Starting training for {self.args.epochs} epochs...")
        training_interrupted = False

        for epoch in range(1, self.args.epochs + 1):
            total_epoch_loss = 0.0
            batch_loss_sums = defaultdict(float)
            self.model.train()

            for i, batch_data in enumerate(data_loader):
                bow_lang1 = batch_data['bow_lang1'].to(self.device, non_blocking=True)
                bow_lang2 = batch_data['bow_lang2'].to(self.device, non_blocking=True)
                cluster_info = batch_data.get('cluster_info', None)
                if isinstance(cluster_info, torch.Tensor):
                    cluster_info = cluster_info.tolist()

                model_outputs = self.model(bow_lang1, bow_lang2, cluster_info)
                current_batch_loss = model_outputs['total_loss']

                if torch.isnan(current_batch_loss) or torch.isinf(current_batch_loss):
                    print(f"\nError: Loss is NaN or Inf at Epoch {epoch}, Batch {i+1}. Stopping training.")
                    training_interrupted = True
                    break

                optimizer.zero_grad()
                current_batch_loss.backward()
                optimizer.step()

                total_epoch_loss += current_batch_loss.item() * len(bow_lang1)
                for key in ['tm_loss_lang1', 'tm_loss_lang2', 'contrast_loss', 'loss_MI']:
                    if key in model_outputs and isinstance(model_outputs[key], torch.Tensor):
                        batch_loss_sums[key] += model_outputs[key].item()
                    elif key in model_outputs:
                        batch_loss_sums[key] += float(model_outputs[key])

            if training_interrupted:
                break

            if lr_scheduler:
                lr_scheduler.step()

            avg_total_loss_epoch = total_epoch_loss / data_size if data_size > 0 else 0
            avg_tm_loss_lang1 = batch_loss_sums['tm_loss_lang1'] / num_batch if num_batch > 0 else 0
            avg_tm_loss_lang2 = batch_loss_sums['tm_loss_lang2'] / num_batch if num_batch > 0 else 0
            avg_contrast_loss = batch_loss_sums['contrast_loss'] / num_batch if num_batch > 0 else 0
            avg_loss_MI = batch_loss_sums['loss_MI'] / num_batch if num_batch > 0 else 0
            avg_combined_tm_loss = avg_tm_loss_lang1 + avg_tm_loss_lang2

            epoch_detailed_losses = {
                'epoch': epoch,
                'total_loss': avg_total_loss_epoch,
                'tm_loss_lang1': avg_tm_loss_lang1,
                'tm_loss_lang2': avg_tm_loss_lang2,
                'contrast_loss': avg_contrast_loss,
                'loss_MI': avg_loss_MI
            }
            all_epoch_losses.append(epoch_detailed_losses)

            if self.log_file_path:
                with open(self.log_file_path, 'a') as f:
                    f.write(json.dumps(epoch_detailed_losses) + '\n')

            print(f"Epoch: {epoch:03d}, Total Loss: {avg_total_loss_epoch:.3f}, "
                  f"TM Loss: {avg_combined_tm_loss:.3f}, "
                  f"Contrast Loss: {avg_contrast_loss:.3f}, "
                  f"MI Loss: {avg_loss_MI:.3f}")

            if hasattr(self.args, 'output_dir') and (epoch % save_frequency == 0 or epoch == self.args.epochs):
                checkpoint_filename = f'model_epoch_{epoch}.pth'
                checkpoint_path = os.path.join(self.args.output_dir, checkpoint_filename)
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  Model state saved to {checkpoint_path}")

        print("Training finished.")

        beta_lang1_np, beta_lang2_np = None, None
        if not training_interrupted:
            beta_lang1_tensor, beta_lang2_tensor = self.model.get_beta()
            beta_lang1_np = beta_lang1_tensor.detach().cpu().numpy()
            beta_lang2_np = beta_lang2_tensor.detach().cpu().numpy()

            final_save_data = {
                'model_state_dict': self.model.state_dict(),
                f'beta_{self.lang1}': beta_lang1_np,
                f'beta_{self.lang2}': beta_lang2_np,
                'config_args': vars(self.args)
            }

            if hasattr(self.args, 'output_dir'):
                final_save_filename = 'final_model_state_and_betas.pth'
                final_save_path = os.path.join(self.args.output_dir, final_save_filename)
                torch.save(final_save_data, final_save_path)

        return {
            'beta_lang1': beta_lang1_np,
            'beta_lang2': beta_lang2_np,
            'losses': all_epoch_losses,
            'training_successful': not training_interrupted
        }

    def get_theta(self, data_loader, lang):
        self.model.eval()
        theta_list = []
        internal_loader = DataLoader(
            data_loader.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=data_loader.collate_fn
        )
        
        with torch.no_grad():
            for batch_data in internal_loader:
                bow_key = f'bow_{lang}'
                
                if bow_key not in batch_data:
                    if lang == self.lang1:
                        bow_key = 'bow_lang1'
                    elif lang == self.lang2:
                        bow_key = 'bow_lang2'
                    else:
                        for key in batch_data:
                            if key.startswith('bow_'):
                                bow_key = key
                                break
                
                bow = batch_data[bow_key].to(self.device, non_blocking=True)
                
                if lang == self.lang1:
                    encoder_method = self.model.encoder_lang1
                else:
                    encoder_method = self.model.encoder_lang2
                
                theta_batch, _, _ = encoder_method(bow)
                theta_list.append(theta_batch.detach().cpu().numpy())

        if not theta_list:
            return np.array([])
        return np.concatenate(theta_list, axis=0)

    def test(self, data_loader):
        theta_lang1 = self.get_theta(data_loader, self.lang1)
        theta_lang2 = self.get_theta(data_loader, self.lang2)
        return theta_lang1, theta_lang2