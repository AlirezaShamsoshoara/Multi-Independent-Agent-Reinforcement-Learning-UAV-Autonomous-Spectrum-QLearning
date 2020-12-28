"""
#################################
# Plot location based on the updated dictionary or location data
#################################
"""

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt


#########################################################
# Function definition


def reopen_figure(fig):
    # create a dummy figure and use its
    # manager to display "fig"

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show(block=False)
    return dummy, fig


def update3d_figure(prev_fig, location, height, width):
    x_u = location.get('X_U')
    x_s = location.get('X_S')
    x_f = location.get('X_F')
    x_gt = location.get('X_GT')
    x_gr = location.get('X_GR')

    y_u = location.get('Y_U')
    y_s = location.get('Y_S')
    y_f = location.get('Y_F')
    y_gt = location.get('Y_GT')
    y_gr = location.get('Y_GR')

    z_u = location.get('Z_U')
    z_s = location.get('Z_S')
    z_f = location.get('Z_F')
    z_gt = location.get('Z_GT')
    z_gr = location.get('Z_GR')

    if prev_fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = prev_fig.add_subplot(111, projection='3d')

    ax.plot(np.squeeze([x_s, x_f]), np.squeeze([y_s, y_f]), np.squeeze([z_s, z_f]), 'ro', markersize=12)
    # ax.axis([x_s - 10, 10 + x_f, 0-10, Width + 10, 0 - 10, Height + 10])
    ax.set_xlim(x_s - 10, 10 + x_f)
    ax.set_ylim(0 - 10, width + 10)
    ax.set_zlim(0 - 10, height + 10)
    ax.set_xlabel('X axis (L)')
    ax.set_ylabel('Y axis (W)')
    ax.set_zlabel('Z axis (H)')
    ax.grid(True)
    plt.show(block=False)
    # plt.arrow(x_s-5, y_s, x_s, y_s, width=1)

    ax.plot(np.squeeze([x_gt, x_gr]), np.squeeze([y_gt, y_gr]), np.squeeze([z_gt, z_gr]), 'bo', markersize=15)

    ax.plot(x_u[:, 0], y_u[:, 0], z_u[:, 0], 'go', markersize=10)
    k = 0
    for i, j, l in zip(x_u[:, 0], y_u[:, 0], z_u[:, 0]):
        corr = -0.05  # adds a little correction to put annotation in marker's centrum
        # ax.annotate(str(k), xyz=(i + corr, j + corr, l + corr))
        ax.text(i, j, l, '%s' % str(k))
        k += 1
    # rect = Rectangle((0, 0), Length, Width, fill=0, alpha=1)
    # ax.add_patch(rect)
    # art3d.pathpatch_2d_to_3d(rect, z=0, zdir="x")

    plt.show(block=False)
    return prev_fig


def update2d_figure(prev_fig, location, width, region_length):
    x_u = location.get('X_U')
    x_s = location.get('X_S')
    x_f = location.get('X_F')
    x_gt = location.get('X_GT')
    x_gr = location.get('X_GR')

    y_u = location.get('Y_U')
    y_s = location.get('Y_S')
    y_f = location.get('Y_F')
    y_gt = location.get('Y_GT')
    y_gr = location.get('Y_GR')

    if prev_fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = prev_fig.add_subplot(111)

    ax.plot(np.squeeze([x_s, x_f]), np.squeeze([y_s, y_f]), 'ro', markersize=12)

    ax.set_xlim(x_s - 2, 2 + x_f)
    ax.set_ylim(0 - 2, width + 2)

    ax.set_xlabel('X axis (L)')
    ax.set_ylabel('Y axis (W)')

    # ax.grid(True)
    # ax.grid(which='minor', alpha=0.2)
    # ax.grid(which='major', alpha=0.5)

    major_ticks_x = np.arange(0 - .5, width + 0.5, region_length)
    minor_ticks_x = np.arange(x_s - 1.5, x_f + 0.5, 1)

    major_ticks_y = np.arange(0-0.5, width + 1.5, region_length)
    minor_ticks_y = np.arange(0-0.5, width + 1.5, 1)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1.0)

    plt.show(block=False)

    ax.plot(np.squeeze([x_gt, x_gr]), np.squeeze([y_gt, y_gr]), 'bo', markersize=15)

    ax.plot(x_u[:, 0], y_u[:, 0], 'go', markersize=10)
    k = 0
    for i, j in zip(x_u[:, 0], y_u[:, 0]):
        corr = -0.05  # adds a little correction to put annotation in marker's centrum
        # ax.annotate(str(k), xyz=(i + corr, j + corr, l + corr))
        ax.text(i, j, '%s' % str(k))
        k += 1

    plt.show(block=False)
    return prev_fig
