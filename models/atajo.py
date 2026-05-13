import torch

try:
    # Añadimos weights_only=False para decirle a PyTorch que el archivo es seguro
    ckpt = torch.load('best.pt', map_location='cpu', weights_only=False)
    
    if 'train_args' in ckpt:
        args = ckpt['train_args']
        print(f"Dataset (data): {args.get('data')}")
        print(f"Proyecto/Nombre: {args.get('project')}/{args.get('name')}")
        
    if 'model' in ckpt:
        modelo = ckpt['model']
        if hasattr(modelo, 'names'):
            print(f"Clases: {modelo.names}")
            
except Exception as e:
    print(f"Error al cargar: {e}")
