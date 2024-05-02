import torch
from torch.nn import KLDivLoss, CrossEntropyLoss
import torch.optim as optim
from myModel import EntityModel
from transformers import AutoModel


# Path to the pre-trained model
model_path = "../../models/entity_model_distilbert-base-uncased_20240204_221018_lr2e-05_dropout0_batch16_seed123.pth"
path_parts = model_path.split("_")
model_name = path_parts[1]
model_name = 'distilbert-base-uncased'

# Initialize the model
teacher_model = EntityModel(model_name=model_name, num_labels=4, dropout_rate=0)
teacher_model.load_state_dict(torch.load(model_path))
teacher_model.eval()

# Initialize the Student Model
student_model_name = "prajjwal1/bert-mini"  # Replace with your desired student model
student_model = AutoModel.from_pretrained(student_model_name)  # Initialize your custom EntityModel or adjust accordingly

# Knowledge Distillation Training
distillation_loss_fn = KLDivLoss(reduction='batchmean')
classification_loss_fn = CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=5e-5)

print(teacher_model, student_model)

# Distillation Training Loop
num_epochs = 5  # Adjust as needed
for epoch in range(num_epochs):
    for inputs, labels in dataloader:  # Assuming a dataloader is available
        # Forward pass of teacher with input
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Forward pass of student
        student_outputs = student_model(inputs)

        # Calculate loss
        loss = distillation_loss_fn(student_outputs, teacher_outputs) + \
               classification_loss_fn(student_outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the distilled student model
student_model.save_pretrained('path_to_save_student_model')
