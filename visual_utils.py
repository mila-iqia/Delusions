"""
Mostly from Skipper
"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from PIL import Image
import matplotlib.patches as patches
import cv2
from utils import minigridobs2tensor, code2idx
import torch
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from math import acos, degrees
from matplotlib.pyplot import cm

def plot_EM(x, h, size_batch, num_classes, n_stdev_bands=3, prior=None):
    plt. clf()
    variance, transform = torch.linalg.eigh(h.covariance_matrix)
    stdev = variance.sqrt()
    ax = plt.subplot(111, aspect='equal')

    llhood = h.log_prob(x.unsqueeze(1).to(h.mean.device))
    if prior is None:
        weighted_llhood = llhood
    else:
        weighted_llhood = llhood + prior
    log_sum_lhood = torch.logsumexp(weighted_llhood, dim=1, keepdim=True)
    log_posterior = weighted_llhood - log_sum_lhood

    pi = torch.exp(log_posterior.reshape(x.shape[0], num_classes, 1))
    classes = pi.reshape(size_batch, num_classes).argmax(-1).cpu()

    legend = []
    cmap = cm.rainbow(torch.linspace(0, 1, h.mean.size(0)))
    idx = 0
    for mean, stdev, transform, color in zip(h.mean, stdev, transform, cmap):
        legend += [mpatches.Patch(color=color, label=f'mu {mean[0].item():.2f}, {mean[1].item():.2f} '
                                                    f'sigma {stdev[0].item():.2f} {stdev[1].item():.2f}')]
        for j in range(1, n_stdev_bands+1):
            ell = Ellipse(xy=(mean[0], mean[1]),
                        width=stdev[0] * j * 2, height=stdev[1] * j * 2,
                        angle=degrees(acos(transform[0, 0].item())),
                        alpha=1.0,
                        edgecolor=color,
                        fc='none')
            ax.add_artist(ell)
        x_class = x[classes == idx]
        plt.scatter(x_class[:, 0], x_class[:, 1], color=color)
        idx += 1
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(handles=legend)
    plt.draw()
    plt.pause(0.0001)

def classes_ratio(writer, step_record, encoder_state, encoder_situation, obses, num_categoricals, num_categories, indices_reachable=None):
    states = encoder_state(obses)
    situations = encoder_situation(states)
    ind_situations = code2idx(situations, num_categoricals, num_categories).cpu().numpy()
    if indices_reachable is None:
        num_situations = int(ind_situations.max()) + 1
        for i in range(num_situations):
            writer.add_scalar(f"Preds/class_distribution/ratio_{i}", (ind_situations == i).sum() / obses.shape[0], step_record)
    else:
        ind_situations = ind_situations[indices_reachable]
        num_situations = int(ind_situations.max()) + 1
        for i in range(num_situations):
            writer.add_scalar(f"Preds/class_distribution_reachable/ratio_{i}", (ind_situations == i).sum() / len(indices_reachable), step_record)

@torch.no_grad()
def visualize_classes(writer, step_record, encoder_state, encoder_situation, func_env, num_categoricals, num_categories, size_world, suffix="", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    coor_i, coor_j = np.meshgrid(np.arange(size_world), np.arange(size_world), indexing='ij')
    coor_d = np.zeros_like(coor_i)
    coor_i, coor_j, coor_d = coor_i.reshape(-1), coor_j.reshape(-1), coor_d.reshape(-1)
    env = func_env()
    env.reset()
    obses = env.draw_obs_with_agent(coor_i, coor_j, coor_d)
    states = encoder_state(minigridobs2tensor(obses, device=device))
    ind_situations = code2idx(encoder_situation(states), num_categoricals, num_categories).cpu().numpy()
    background_rendered = env.render_image()
    plt.axis("off")
    plt.margins(0)
    def get_img_from_fig(fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def discrete_matshow(data, vmin, vmax):
        cmap = plt.get_cmap('RdBu', vmax - vmin + 1)
        plt.matshow(np.flip(data.T, 0), fignum=0, cmap=cmap, vmin=vmin - 0.5, vmax=vmax + 0.5)
        rgb_array = get_img_from_fig(plt.gcf(), 100)
        plt.close()
        return rgb_array
    mask_rendered = discrete_matshow(ind_situations.reshape(size_world, size_world), vmin=0, vmax=num_categories ** num_categoricals - 1)
    mask_rendered = cv2.resize(mask_rendered, (background_rendered.shape[0], background_rendered.shape[1]), interpolation=cv2.INTER_NEAREST)
    combined = cv2.addWeighted(background_rendered, 0.5, mask_rendered, 0.5, 0)
    writer.add_image(f"classes{suffix}/combined", combined, step_record, dataformats="HWC")
    writer.add_image(f"classes{suffix}/mask", mask_rendered, step_record, dataformats="HWC")
    writer.flush()
    return combined

@torch.no_grad()
def visualize_classes_SSM(writer, step_record, encoder_state, encoder_situation, func_env, num_categoricals, num_categories, size_world, suffix="", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    env = func_env()
    env.reset()
    env.collect_states_reachable()
    states = encoder_state(minigridobs2tensor(env.state2obs(np.arange(env.DP_info["num_states"])), device=device))
    ind_situations = code2idx(encoder_situation(states), num_categoricals, num_categories).cpu().numpy().reshape(4, size_world, size_world)

    def get_img_from_fig(fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def discrete_matshow(data, vmin, vmax):
        cmap = plt.get_cmap('RdBu', vmax - vmin + 1)
        plt.matshow(np.flip(data.T, 0), fignum=0, cmap=cmap, vmin=vmin - 0.5, vmax=vmax + 0.5)
        rgb_array = get_img_from_fig(plt.gcf(), 100)
        plt.close()
        return rgb_array

    for x in range(4):
        background_rendered = env.render_image(obs=env.ijxd2obs(env.pos_agent_init[0], env.pos_agent_init[1], x, 0))
        plt.axis("off")
        plt.margins(0)
    
        mask_rendered = discrete_matshow(ind_situations[x, :, :], vmin=0, vmax=num_categories ** num_categoricals - 1)
        mask_rendered = cv2.resize(mask_rendered, (background_rendered.shape[0], background_rendered.shape[1]), interpolation=cv2.INTER_NEAREST)
        combined = cv2.addWeighted(background_rendered, 0.5, mask_rendered, 0.5, 0)
        writer.add_image(f"classes_{x}{suffix}/combined", combined, step_record, dataformats="HWC")
        writer.add_image(f"classes_{x}{suffix}/mask", mask_rendered, step_record, dataformats="HWC")
    writer.flush()
    return combined

def draw_bidir_arrow(ax, i1, j1, i2, j2, annotation="x", color="blue", arrowstyle="<|-|>", size_grid=np.array([32, 32]), size_rendered=[256, 256]):
    ij2xy = lambda i, j: (
        (i + 0.5) * size_grid[0],
        size_rendered[0] - (j + 0.5) * size_grid[1],
    )
    x1, y1 = ij2xy(i1, j1)
    x2, y2 = ij2xy(i2, j2)
    p1 = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=arrowstyle, mutation_scale=20, color=color)
    ax.add_patch(p1)
    ax.text(0.5 * (x1 + x2), 0.5 * (y1 + y2), annotation, color=color, fontsize=12)


def visualize_waypoint_graph(rendered, aux, env, annotation="reward", alpha=0.5, dist_cutoff=500):
    num_waypoints = aux["distances"].shape[0]
    my_dpi = 100
    fig = plt.figure(figsize=(rendered.shape[1] / my_dpi, rendered.shape[0] / my_dpi), dpi=my_dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim([0, rendered.shape[1]])
    ax.set_ylim([0, rendered.shape[0]])
    ax.axis("off")
    ax.margins(0)
    ax.imshow(rendered, alpha=alpha)
    for i in range(num_waypoints):
        for j in range(num_waypoints):
            if i == j:
                continue
            if float(min(aux["distances"][i, j], aux["distances"][j, i])) < dist_cutoff:
                if annotation in ["Q"]:
                    if j != num_waypoints - 1 and i != 0:
                        continue
                    anno = ("%.2g" % max(aux[annotation][i, j], aux[annotation][j, i])).lstrip("0")
                elif annotation in ["distances"]:
                    anno = "%.2g" % min(aux[annotation][i, j], aux[annotation][j, i])
                else:
                    value = float(max(aux[annotation][i, j], aux[annotation][j, i]))
                    if value < 1e-3:
                        continue
                    anno = ("%.2g" % value).lstrip("0")
                if float(aux["distances"][j, i]) > dist_cutoff:
                    arrowstyle = "-|>"
                    color = "red"
                elif float(aux["distances"][i, j]) > dist_cutoff:
                    arrowstyle = "<|-"
                    color = "red"
                else:
                    arrowstyle = "<|-|>"
                    color = "blue"
                if env.name_game == "RandDistShift":
                    draw_bidir_arrow(
                        ax,
                        aux["ijxds"][i, 0],
                        aux["ijxds"][i, 1],
                        aux["ijxds"][j, 0],
                        aux["ijxds"][j, 1],
                        annotation=anno,
                        color=color,
                        arrowstyle=arrowstyle,
                        size_rendered=rendered.shape,
                    )
                elif env.name_game == "SwordShieldMonster":
                    draw_bidir_arrow(
                        ax,
                        aux["ijxds"][i, 0] + env.width * aux["ijxds"][i, 2],
                        aux["ijxds"][i, 1],
                        aux["ijxds"][j, 0] + env.width * aux["ijxds"][j, 2],
                        aux["ijxds"][j, 1],
                        annotation=anno,
                        color=color,
                        arrowstyle=arrowstyle,
                        size_rendered=rendered.shape,
                    )

    fig.canvas.draw()
    rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return rgb_array

def visualize_plan(rendered, aux, q, env, alpha=0.5):
    num_waypoints = aux["distances"].shape[0]
    my_dpi = 100
    fig = plt.figure(figsize=(rendered.shape[1] / my_dpi, rendered.shape[0] / my_dpi), dpi=my_dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim([0, rendered.shape[1]])
    ax.set_ylim([0, rendered.shape[0]])
    ax.axis("off")
    ax.margins(0)
    ax.imshow(rendered, alpha=alpha)
    idx_waypoint_next = 0
    picked = np.zeros(num_waypoints, dtype=bool)
    while not picked[idx_waypoint_next] and idx_waypoint_next != num_waypoints - 1:
        picked[idx_waypoint_next] = True
        i = idx_waypoint_next
        idx_waypoint_next = q[i, :].argmax()
        ijx_next = aux["ijxds"][idx_waypoint_next, :]
        ijx_curr = aux["ijxds"][i, :]
        arrowstyle = "-|>"
        color = "green"
        value = q[i, idx_waypoint_next].item()
        to_disp = ("%.2g" % (value,)).lstrip("0")
        if env.name_game == "RandDistShift":
            draw_bidir_arrow(
            ax, ijx_curr[0], ijx_curr[1], ijx_next[0], ijx_next[1], annotation=to_disp, color=color, arrowstyle=arrowstyle, size_rendered=rendered.shape
            )
        elif env.name_game == "SwordShieldMonster":
            draw_bidir_arrow(
                ax, ijx_curr[0] + env.width * ijx_curr[2], ijx_curr[1], ijx_next[0] + env.width * ijx_next[2], ijx_next[1], annotation=to_disp, color=color, arrowstyle=arrowstyle, size_rendered=rendered.shape
            )
    fig.canvas.draw()
    rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return rgb_array

def gen_comparative_image(images_gen, image_base):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(8, 8))
    for i in range(len(images_gen)):
        if i >= 64:
            break
        ax = plt.subplot(8, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_gen[i])
        ax.set_aspect("equal")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    buf = io.BytesIO()
    plt.margins(0, 0)
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img_gen = np.asarray(img)[:, :, :3]
    plt.close()

    figure = plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_base = Image.fromarray(image_base)
    image_base = image_base.resize((image_base.size[0] * 8, image_base.size[1] * 8), Image.Resampling.LANCZOS)
    plt.imshow(image_base)
    buf = io.BytesIO()
    plt.margins(0, 0)
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img_base = np.asarray(img)[:, :, :3]
    plt.close()

    img_cat = np.concatenate((img_base, img_gen[:, 2:-1, :]), axis=0)
    return img_cat.transpose(2, 1, 0)


def outline(image, color="red", margin=10):
    assert color in ["red", "blue", "green"]
    if color == "red":
        target_channel = 0
    elif color == "blue":
        target_channel = 2
    elif color == "green":
        target_channel = 1
    image_ = np.copy(image)

    image_[:margin, :, :] = 0
    image_[:margin, :, target_channel] = 255
    image_[-margin:, :, :] = 0
    image_[-margin:, :, target_channel] = 255
    image_[:, :margin, :] = 0
    image_[:, :margin, target_channel] = 255
    image_[:, -margin:, :] = 0
    image_[:, -margin:, target_channel] = 255
    return image_
