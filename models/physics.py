import torch
import torch.nn.functional as F

###########################################################
# Compute gradients
def grad(tensor, dx, dy):
    gradient_x = torch.full_like(tensor, float('nan'))
    gradient_y = torch.full_like(tensor, float('nan'))
    
    gradient_x[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / (2 * dx)
    gradient_y[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / (2 * dy)
    
    gradient_x[:, 0] = (tensor[:, 1] - tensor[:, 0]) / dx
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / dx
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / dy
    gradient_y[-1, :] = (tensor[-1] - tensor[-2, :]) / dy

    return gradient_x, gradient_y

###########################################################
# Compute higher order gradients
def ho_grad(tensor, dx, dy):
    gradient_x = torch.full_like(tensor, float('nan'))
    gradient_y = torch.full_like(tensor, float('nan'))

    gradient_x[:, 2:-2] = (-tensor[:, 4:] + 8*tensor[:, 3:-1] - 8*tensor[:, 1:-3] + tensor[:, :-4]) / (12 * dx)
    gradient_y[2:-2, :] = (-tensor[4:, :] + 8*tensor[3:-1, :] - 8*tensor[1:-3, :] + tensor[:-4, :]) / (12 * dy)

    gradient_x[:, 0] = (tensor[:, 1] - tensor[:, 0]) / dx
    gradient_x[:, 1] = (tensor[:, 2] - tensor[:, 0]) / (2 * dx)
    gradient_x[:, -1] = (tensor[:, -1] - tensor[:, -2]) / dx
    gradient_x[:, -2] = (tensor[:, -1] - tensor[:, -3]) / (2 * dx)
    gradient_y[0, :] = (tensor[1, :] - tensor[0, :]) / dy
    gradient_y[1, :] = (tensor[2, :] - tensor[0, :]) / (2 * dy)
    gradient_y[-1, :] = (tensor[-1] - tensor[-2, :]) / dy
    gradient_y[-2, :] = (tensor[-1] - tensor[-3, :]) / (2 * dy)

    return gradient_x, gradient_y

###########################################################    
def loss_huber(residuals, delta=1.0):
    residuals_flattened = residuals.view(-1)
    loss = F.huber_loss(residuals_flattened, torch.zeros_like(residuals_flattened), delta=delta)
    return loss

###########################################################
def continuity_h(inputs, preds, dx, dy, delta=1.0):
    U = inputs[2, :, :].squeeze()
    V = inputs[3, :, :].squeeze()
    h = -1 * preds[0, :, :]  # Elevation to depth

    hU = h*U
    hV = h*V

    hUx, _ = ho_grad(hU, dx, dy)
    _, hVy = ho_grad(hV, dx, dy)
    
    valid_mask = ~torch.isnan(hUx) & ~torch.isnan(hVy) 
    hUx, hVy = hUx[valid_mask], hVy[valid_mask]

    residuals = hUx + hVy

    loss = loss_huber(residuals, delta)
    return loss

###########################################################
def continuity_h2(inputs, preds, dx, dy, delta=1.0):
    U = inputs[2, :, :].squeeze()
    V = inputs[3, :, :].squeeze()
    h = -1 * preds[0, :, :]  # Elevation to depth

    hU = h*U
    hV = h*V
    hUU = hU*U
    hUV = hU*V
    hVV = hV*V

    hx, hy = ho_grad(h, dx, dy)
    hUx, _ = ho_grad(hU, dx, dy)
    _, hVy = ho_grad(hV, dx, dy)
    hUUx, _ = ho_grad(hUU, dx, dy)
    hUVx, hUVy = ho_grad(hUV, dx, dy)
    _, hVVy = ho_grad(hVV, dx, dy)
    
    valid_mask = ~torch.isnan(hx) & ~torch.isnan(hy) & ~torch.isnan(hUx) & ~torch.isnan(hVy) & ~torch.isnan(hUUx) & ~torch.isnan(hUVy) & ~torch.isnan(hUVx) & ~torch.isnan(hVVy)
    h = h[valid_mask]
    hx, hy = hx[valid_mask], hy[valid_mask]
    hUx, hVy = hUx[valid_mask], hVy[valid_mask]
    hUUx = hUUx[valid_mask]
    hUVx, hUVy = hUVx[valid_mask], hUVy[valid_mask]
    hVVy = hVVy[valid_mask]

    residual_cont = hUx + hVy
    residual_momX = hUUx + hUVy + 9.81*h*hx
    residual_momY = hUVx + hVVy + 9.81*h*hy
    residuals = torch.cat((residual_cont, residual_momX, residual_momY))

    loss = loss_huber(residuals, delta)
    return loss

###########################################################
def continuity_uv(inputs, preds, dx, dy, delta=1.0):
    U = preds[0, :, :].squeeze()
    V = preds[1, :, :].squeeze()
    x_size, y_size = U.shape
    y_values = torch.linspace(-2, 4, steps=y_size)
    h = y_values.expand(x_size, -1).to(U.device)

    Ux, Uy = ho_grad(U, dx, dy)
    Vx, Vy = ho_grad(V, dx, dy)
    Uxx, _ = ho_grad(Ux, dx, dy)
    _, Uyy = ho_grad(Uy, dx, dy)
    Vxx, _ = ho_grad(Vx, dx, dy)
    _, Vyy = ho_grad(Vy, dx, dy)
    hx, hy = ho_grad(h, dx, dy)

    nu = 1e-6  # Kinematic viscosity for saltwater in m^2/s
    residual_cont = h * Ux + hx * U + h * Vy + hy * V
    residual_momX = Ux * U + Uy * V - nu * (Uxx + Uyy)
    residual_momY = Vx * U + Vy * V - nu * (Vxx + Vyy)
    residuals = torch.cat((residual_cont, residual_momX, residual_momY))

    loss = loss_huber(residuals, delta)
    return loss