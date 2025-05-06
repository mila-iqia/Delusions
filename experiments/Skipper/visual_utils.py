import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from PIL import Image
import matplotlib.patches as patches
import torch
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from math import acos, degrees
from matplotlib.pyplot import cm

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
