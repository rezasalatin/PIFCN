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
def compute_gradients(tensor, spacing=0.1):
    # Initialize gradient tensors
    gradient_x = torch.zeros_like(tensor)
    gradient_y = torch.zeros_like(tensor)
    
    # Centered differences for interior points
    gradient_x[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / (2 * spacing)
    gradient_y[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / (2 * spacing)
    
    # One-way differences for edges
    # For x
    gradient_x[:, 0] = (tensor[:, 1] - tensor[:, 0]) / spacing  # Forward difference
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / spacing  # Backward difference
    
    # For y
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / spacing  # Forward difference
    gradient_y[-1, :] = (tensor[-1, :] - tensor[-2, :]) / spacing  # Backward difference

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
def continuity_only(inputs, targets):
    
    U = inputs[0, :, :].squeeze()
    V = inputs[1, :, :].squeeze()
    h = targets[:, :]

    #hx, hy = compute_autograd(h, x), compute_autograd(h, y)
    hy, hx = compute_gradients(h, spacing=0.1)
    Uy, Ux = compute_gradients(U, spacing=0.1)
    Vy, Vx = compute_gradients(V, spacing=0.1)
    
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
    #fc = torch.mean((hx*U+Ux*h + hy*V+Vy*h)**2)
    fc = continuity_equation_loss_huber(U, V, Ux, Uy, Vx, Vy, h, hx, hy, delta=1.0)
    
    # Enforce strict constraint for hx and hy not to exceed absolute values of 0.5 or 0.1
    constraint_violation_hx = torch.max(torch.zeros_like(hx), torch.abs(hx) - 0.5)
    constraint_violation_hy = torch.max(torch.zeros_like(hy), torch.abs(hy) - 0.5)
    # Enforce constraint that h should not be negative
    constraint_violation_h = torch.max(torch.zeros_like(h), - (h + 0.1))
    
    # Polynomial penalty (quadratic for example)
    penalty_h = (constraint_violation_hx**2) + (constraint_violation_hy**2) + (constraint_violation_h**2)

    # Apply an exponential penalty for any violation
    #penalty_h = torch.exp(constraint_violation_hx) + torch.exp(constraint_violation_hy) + torch.exp(constraint_violation_h) - 3  # Subtract 0.6 to offset the base case where there's no violation

    # Total loss
    loss = fc #+ penalty_h.sum()

    return loss
