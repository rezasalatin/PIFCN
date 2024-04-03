import torch
import torch.nn.functional as F

###########################################################
# compute gradients
def compute_gradients(tensor, dx, dy):
    # Initialize gradient tensors with NaNs
    gradient_x = torch.full_like(tensor, float('nan'))
    gradient_y = torch.full_like(tensor, float('nan'))
    
    # Centered differences for interior points
    gradient_x[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / (2 * dx)
    gradient_y[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / (2 * dy)
    
    # One-way differences for edges
    # For x
    gradient_x[:, 0] = (tensor[:, 1] - tensor[:, 0]) / dx  # Forward difference
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / dx  # Backward difference, use dx instead of dy for consistency
    
    # For y
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / dy  # Forward difference
    gradient_y[-1, :] = (tensor[-1, :] - tensor[-2, :]) / dy  # Backward difference

    return gradient_y, gradient_x

###########################################################    
def continuity_equation_loss_huber(U, V, Ux, Uy, Vx, Vy, h, hx, hy, delta):
    # Calculate the residuals for the continuity equation
    residuals = (hx*U + Ux*h + hy*V + Vy*h)
    
    # Flatten the residuals to use with huber_loss, which expects 1D inputs
    residuals_flattened = residuals.view(-1)
    
    # Calculate loss for the residuals
    loss = F.huber_loss(residuals_flattened, torch.zeros_like(residuals_flattened), delta=delta)
    #loss = F.mse_loss(residuals_flattened, torch.zeros_like(residuals_flattened))

    return loss

###########################################################
def continuity_only(inputs, targets, dx, dy, delta=1.0):
    
    U = inputs[0, :, :].squeeze()
    V = inputs[1, :, :].squeeze()
    h = targets[:, :]
    h = -h

    #hx, hy = compute_autograd(h, x), compute_autograd(h, y)
    hy, hx = compute_gradients(h, dx, dy)
    Uy, Ux = compute_gradients(U, dx, dy)
    Vy, Vx = compute_gradients(V, dx, dy)

    # Create a mask for non-NaN values
    valid_mask = ~torch.isnan(h) & \
            ~torch.isnan(U) & ~torch.isnan(V) & \
            ~torch.isnan(Ux) & ~torch.isnan(Vy)
            
    # Apply the mask to all variables
    U, V = U[valid_mask], V[valid_mask]
    Ux = Ux[valid_mask]
    Vy = Vy[valid_mask]
    h, hx, hy = h[valid_mask], hx[valid_mask], hy[valid_mask]

    # Continuity equation loss
    loss = continuity_equation_loss_huber(U, V, Ux, Uy, Vx, Vy, h, hx, hy, delta)

    return loss
