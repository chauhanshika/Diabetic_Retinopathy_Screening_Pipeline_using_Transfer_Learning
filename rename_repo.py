import os

# Update README.md
readme_path = 'README.md'
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("Diabetic-Retinopathy-Using-Quantum-Transfer-Learning", "Diabetic-Retinopathy-Screening-Pipeline")
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
artifacts_dir = r"C:\\Users\\chauh\\.gemini\\antigravity\\brain\\62da9219-a774-4d85-b839-2b93541f21bb"

# Update task.md
task_path = os.path.join(artifacts_dir, 'task.md')
if os.path.exists(task_path):
    with open(task_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("# Diabetic Retinopathy Classification Project Upgrade", "# Diabetic Retinopathy Screening Pipeline using Transfer Learning")
    
    with open(task_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Update implementation_plan.md
impl_path = os.path.join(artifacts_dir, 'implementation_plan.md')
if os.path.exists(impl_path):
    with open(impl_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("# Upgrading Diabetic Retinopathy Classification Project", "# Diabetic Retinopathy Screening Pipeline using Transfer Learning")
    
    with open(impl_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Renaming scripts completed successfully.")
