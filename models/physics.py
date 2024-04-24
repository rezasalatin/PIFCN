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
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / dx  # Backward difference
    # For y
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / dy  # Forward difference
    gradient_y[-1, :] = (tensor[-1, :] - tensor[-2, :]) / dy  # Backward difference

    return gradient_x, gradient_y

###########################################################
# compute higher order gradients
def compute_higher_order_gradients(tensor, dx, dy):
    # Initialize gradient tensors with NaNs
    gradient_x = torch.full_like(tensor, float('nan'))
    gradient_y = torch.full_like(tensor, float('nan'))

    # Fourth-order centered differences for interior points
    gradient_x[:, 2:-2] = (-tensor[:, 4:] + 8*tensor[:, 3:-1] - 8*tensor[:, 1:-3] + tensor[:, :-4]) / (12 * dx)
    gradient_y[2:-2, :] = (-tensor[4:, :] + 8*tensor[3:-1, :] - 8*tensor[1:-3, :] + tensor[:-4, :]) / (12 * dy)

    # Simplified one-way differences for edges (more complex schemes could be used here too)
    gradient_x[:, 0] = (tensor[:, 1] - tensor[:, 0]) / dx  # Forward difference
    gradient_x[:, 1] = (tensor[:, 2] - tensor[:, 0]) / (2 * dx)  # Second order forward
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / dx  # Backward difference
    gradient_x[:, -2] = (tensor[:, -1] - tensor[:, -3]) / (2 * dx)  # Second order backward
    # For y
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / dy  # Forward difference
    gradient_y[1, :] = (tensor[2, :] - tensor[0, :]) / (2 * dy)  # Second order forward
    gradient_y[-1, :] = (tensor[-1, :] - tensor[-2, :]) / dy  # Backward difference
    gradient_y[-2, :] = (tensor[-1, :] - tensor[-3, :]) / (2 * dy)  # Second order backward

    return gradient_x, gradient_y

###########################################################    
def continuity_equation_loss_huber(residuals, delta):
    
    # Flatten the residuals to use with huber_loss, which expects 1D inputs
    residuals_flattened = residuals.view(-1)
    
    # Calculate loss for the residuals
    #loss = F.huber_loss(residuals_flattened, torch.zeros_like(residuals_flattened), delta=delta)
    loss = F.mse_loss(residuals_flattened, torch.zeros_like(residuals_flattened))

    return loss

###########################################################
def continuity_depthintegrated(inputs, targets, dx, dy, delta=1.0):
    
    U = inputs[0, :, :].squeeze()
    V = inputs[1, :, :].squeeze()
    h = -targets[:, :]

    #hx, hy = compute_autograd(h, x), compute_autograd(h, y)
    hx, hy = compute_gradients(h, dx, dy)
    Ux, _ = compute_gradients(U, dx, dy)
    _, Vy = compute_gradients(V, dx, dy)

    # Create a mask for non-NaN values
    valid_mask = ~torch.isnan(h) & \
            ~torch.isnan(U) & ~torch.isnan(V) & \
            ~torch.isnan(Ux) & ~torch.isnan(Vy)
            
    # Apply the mask to all variables
    U, V = U[valid_mask], V[valid_mask]
    Ux = Ux[valid_mask]
    Vy = Vy[valid_mask]
    h, hx, hy = h[valid_mask], hx[valid_mask], hy[valid_mask]

    # Calculate the residuals for the continuity equation
    residuals = (hx*U + Ux*h + hy*V + Vy*h)

    # Continuity equation loss
    loss = continuity_equation_loss_huber(residuals, delta)

    return loss

###########################################################
def continuity(inputs, dx, dy, delta=1.0):
    
    U = inputs[0, :, :].squeeze()
    V = inputs[1, :, :].squeeze()

    #hx, hy = compute_autograd(h, x), compute_autograd(h, y)
    Ux, _ = compute_higher_order_gradients(U, dx, dy)
    _, Vy = compute_higher_order_gradients(V, dx, dy)

    # Calculate the residuals for the continuity equation
    residuals = Ux + Vy

    # Continuity equation loss
    loss = continuity_equation_loss_huber(residuals, delta)

    return loss