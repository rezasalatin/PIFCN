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

    Ux, Uy = ho_grad(U, dx, dy)
    Vx, Vy = ho_grad(V, dx, dy)
    Uxx, _ = ho_grad(Ux, dx, dy)
    _, Uyy = ho_grad(Uy, dx, dy)
    Vxx, _ = ho_grad(Vx, dx, dy)
    _, Vyy = ho_grad(Vy, dx, dy)

    hx, hy = ho_grad(h, dx, dy)
    
    valid_mask = ~torch.isnan(h)   
    U, V = U[valid_mask], V[valid_mask]
    Ux, Uy = Ux[valid_mask], Uy[valid_mask]
    Vx, Vy = Vx[valid_mask], Vy[valid_mask]
    Uxx, Uyy = Uxx[valid_mask], Uyy[valid_mask]
    Vxx, Vyy = Vxx[valid_mask], Vyy[valid_mask]
    h, hx, hy = h[valid_mask], hx[valid_mask], hy[valid_mask]

    nu = 1e-6  # Kinematic viscosity for saltwater in m^2/s
    residual_cont = h * Ux + hx * U + h * Vy + hy * V
    residual_momX = Ux * U + Uy * V - nu * (Uxx + Uyy)
    residual_momY = Vx * U + Vy * V - nu * (Vxx + Vyy)
    residuals = torch.cat((residual_cont, residual_momX, residual_momY))

    loss = loss_huber(residuals, delta)
    return loss

###########################################################
def continuity_uv(inputs, preds, dx, dy, delta=1.0):
    U = preds[0, :, :].squeeze()
    V = preds[1, :, :].squeeze()
    h = -1 * inputs[2, :, :]  # Elevation to depth

    Ux, Uy = ho_grad(U, dx, dy)
    Vx, Vy = ho_grad(V, dx, dy)
    Uxx, _ = ho_grad(Ux, dx, dy)
    _, Uyy = ho_grad(Uy, dx, dy)
    Vxx, _ = ho_grad(Vx, dx, dy)
    _, Vyy = ho_grad(Vy, dx, dy)

    x_size, y_size = U.shape
    y_values = torch.linspace(-2, 4, steps=y_size)
    h_firstguess = y_values.expand(x_size, -1).to(U.device)
    hx, hy = ho_grad(h_firstguess, dx, dy)

    nu = 1e-6  # Kinematic viscosity for saltwater in m^2/s
    residual_cont = h * Ux + hx * U + h * Vy + hy * V
    residual_momX = Ux * U + Uy * V - nu * (Uxx + Uyy)
    residual_momY = Vx * U + Vy * V - nu * (Vxx + Vyy)
    residuals = torch.cat((residual_cont, residual_momX, residual_momY))

    loss = loss_huber(residuals, delta)
    return loss

###########################################################
def continuity_all(inputs, dx, dy, delta=1.0):
    U = inputs[0, :, :].squeeze()
    V = inputs[1, :, :].squeeze()
    h = inputs[2, :, :].squeeze()
    h = -h  # Elevation to depth

    Ux, Uy = ho_grad(U, dx, dy)
    Vx, Vy = ho_grad(V, dx, dy)
    hx, hy = ho_grad(h, dx, dy)
    Uxx, _ = ho_grad(Ux, dx, dy)
    _, Uyy = ho_grad(Uy, dx, dy)
    Vxx, _ = ho_grad(Vx, dx, dy)
    _, Vyy = ho_grad(Vy, dx, dy)

    nu = 1e-6  # Kinematic viscosity for saltwater in m^2/s
    residual_cont = (hx * U + Ux * h + hy * V + Vy * h)
    residual_momX = (Ux * U + Uy * V - nu * (Uxx + Uyy))
    residual_momY = (Vx * U + Vy * V - nu * (Vxx + Vyy))

    residuals = torch.cat((residual_cont, residual_momX, residual_momY))

    loss = loss_huber(residuals, delta)
    return loss