import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, Label, Entry, Button
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the Autoencoder model using PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # Assuming data normalization [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model, dataloader, epochs, device, output_text):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        output_text.insert(tk.END, f'Epoch {epoch+1}, Loss: {average_loss:.4f}\n')
    output_text.insert(tk.END, "Training complete!\n")

def save_model(model, output_text):
    if model is None:
        messagebox.showerror("Error", "No model to save.")
        return
    save_path = filedialog.asksaveasfilename(filetypes=[("PyTorch Model", "*.pth")], defaultextension=".pth")
    if save_path:
        torch.save(model.state_dict(), save_path)
        output_text.insert(tk.END, f"Model saved to {save_path}\n")

def load_model(input_dim, encoding_dim, device, output_text):
    model = Autoencoder(input_dim, encoding_dim).to(device)
    load_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
    if load_path:
        model.load_state_dict(torch.load(load_path))
        model.eval()
        output_text.insert(tk.END, "Model loaded successfully.\n")
        return model
    return None

# GUI for controlling the autoencoder
def run_gui(dataframe=None):
    root = tk.Tk()
    root.title("Autoencoder Configuration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    def on_train():
        nonlocal model
        input_dim = int(input_dim_entry.get())
        encoding_dim = int(encoding_dim_entry.get())
        epochs = int(epoch_entry.get())
        batch_size = int(batch_size_entry.get())
        model = Autoencoder(input_dim, encoding_dim).to(device)

        if dataframe is not None:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(dataframe.values)
            # Assume scaled_data is already noisy and the original data is not accessible
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            # Assuming no clean target available, use noisy data as target for unsupervised learning
            dataloader = DataLoader(TensorDataset(data_tensor, data_tensor), batch_size=batch_size, shuffle=True)
            train_autoencoder(model, dataloader, epochs, device, output_text)
        else:
            messagebox.showerror("Error", "No data loaded for training.")


    def on_load_model():
        nonlocal model
        input_dim = int(input_dim_entry.get())
        encoding_dim = int(encoding_dim_entry.get())
        model = load_model(input_dim, encoding_dim, device, output_text)

    def on_predict():
        nonlocal model
        if model is None:
            messagebox.showerror("Error", "Model not trained or initialized.")
            return
        try:
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(dataframe.values)
            noisy_data = data_scaled + 0.1 * np.random.normal(size=data_scaled.shape)
            noisy_data = np.clip(noisy_data, 0, 1)
            input_tensor = torch.tensor(noisy_data, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                predicted = model(input_tensor)
            clean_predicted = scaler.inverse_transform(predicted.cpu().numpy())
            output_text.insert(tk.END, "Denoised data ready. Check Aurora's directory.\n")
            np.savetxt("denoised_data.csv", clean_predicted, delimiter=",")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Layout configuration
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    Label(frame, text="Input Dimension:").pack()
    input_dim_entry = Entry(frame)
    input_dim_entry.pack()

    Label(frame, text="Encoding Dimension:").pack()
    encoding_dim_entry = Entry(frame)
    encoding_dim_entry.pack()

    Label(frame, text="Epochs:").pack()
    epoch_entry = Entry(frame)
    epoch_entry.pack()

    Label(frame, text="Batch Size:").pack()
    batch_size_entry = Entry(frame)
    batch_size_entry.pack()

    Button(frame, text="Train Model", command=on_train).pack()
    Button(frame, text="Load Model", command=on_load_model).pack()

    Button(frame, text="Predict and Save Clean Data", command=on_predict).pack()
    Button(frame, text="Save Model", command=lambda: save_model(model, output_text)).pack()

    output_text = scrolledtext.ScrolledText(frame, height=10)
    output_text.pack(fill=tk.BOTH, expand=True)

    root.mainloop()


def register(app):
    @app.register_plugin('machine_learning', 'ae', 'Denoising Autoencoder')
    def ae():
        dataae = app.get_dataframe()
        run_gui(dataae)
    
