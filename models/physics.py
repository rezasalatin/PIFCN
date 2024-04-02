import torch
import torch.nn.functional as F

###########################################################
# compute gradients for the pinn
def compute_autograd(pred, var):

    grad = torch.autograd.grad(
        pred, var, 
        grad_outputs=torch.ones_like(pred),
        retain_graph=True,
        create_graph=True,
        allow_unused = True
    )[0]

    return grad

###########################################################
# compute gradients
def compute_gradients(tensor, dx, dy):
    # Initialize gradient tensors
    gradient_x = torch.zeros_like(tensor)
    gradient_y = torch.zeros_like(tensor)
    
    # Centered differences for interior points
    gradient_x[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / (2 * dx)
    gradient_y[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / (2 * dy)
    
    # One-way differences for edges
    # For x
    gradient_x[:, 0] = (tensor[:, 1] - tensor[:, 0]) / dx  # Forward difference
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / dy  # Backward difference
    
    # For y
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / dx  # Forward difference
    gradient_y[-1, :] = (tensor[-1, :] - tensor[-2, :]) / dy  # Backward difference

    return gradient_y, gradient_x


###########################################################    
def continuity_equation_loss_huber(U, V, Ux, Uy, Vx, Vy, h, hx, hy, delta=1.0):
    # Calculate the residuals for the continuity equation
    residuals = (hx*U + Ux*h + hy*V + Vy*h)
    
    # Flatten the residuals to use with huber_loss, which expects 1D inputs
    residuals_flattened = residuals.view(-1)
    
    # Calculate Huber loss for the residuals
    # Note: huber_loss expects both 'input' and 'target' arguments,
    # so we use zeros_like to create a target of zeros, indicating we want the residuals to be close to zero.
    loss = F.huber_loss(residuals_flattened, torch.zeros_like(residuals_flattened), delta=delta)
    
    return loss

###########################################################
def continuity_only(inputs, targets, dx, dy):
    
    U = inputs[0, :, :].squeeze()
    V = inputs[1, :, :].squeeze()
    h = targets[:, :]

    #hx, hy = compute_autograd(h, x), compute_autograd(h, y)
    hy, hx = compute_gradients(h, dx, dy)
    Uy, Ux = compute_gradients(U, dx, dy)
    Vy, Vx = compute_gradients(V, dx, dy)
    
    # Create a mask for non-NaN values
    valid_mask = ~torch.isnan(h) & \
            ~torch.isnan(U) & ~torch.isnan(V) & \
            ~torch.isnan(Ux) & ~torch.isnan(Uy) & \
            ~torch.isnan(Vx) & ~torch.isnan(Vy) & \
            ~torch.isnan(hx) & ~torch.isnan(hy)
            
    # Apply the mask to all variables
    U, V = U[valid_mask], V[valid_mask]
    Ux = Ux[valid_mask]
    Vy = Vy[valid_mask]
    h, hx, hy = h[valid_mask], hx[valid_mask], hy[valid_mask]

    # Continuity equation loss
    loss = continuity_equation_loss_huber(U, V, Ux, Uy, Vx, Vy, h, hx, hy, delta=1.0)

    return loss
