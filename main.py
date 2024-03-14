import tkinter as tk
from queue import PriorityQueue, Queue
import heapq

# Define constants for the grid size, cell size, etc.
WIDTH, HEIGHT = 800, 600
ROWS, COLS = 50, 50
CELL_SIZE = WIDTH // COLS

# Define colors and weights
OPEN_COLOR = "white"
WALL_COLOR = "black"
START_COLOR = "green"
END_COLOR = "red"
PATH_COLOR = "blue"
WEIGHT_COLORS = {1: "light gray", 2: "dark gray", 3: "gray"}

# Define the grid
grid = [[1 for _ in range(COLS)] for _ in range(ROWS)]
start = None
end = None


# Heuristic function for A*
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Get neighbors (including diagonal)
def get_neighbors(node, diagonal=False):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if diagonal:
        directions += [(1, 1), (1, -1), (-1, -1), (-1, 1)]

    x, y = node
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < ROWS and 0 <= ny < COLS:
            neighbors.append((nx, ny))
    return neighbors


def a_star_search(start, end):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not pq.empty():
        _, current = pq.get()

        if current == end:
            break

        for next in get_neighbors(
            current, diagonal=True
        ):  # Removed graph parameter, using diagonal
            new_cost = cost_so_far[current] + grid[next[0]][next[1]]  # cost to move
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(end, next)
                pq.put((priority, next))
                came_from[next] = current

    return came_from, cost_so_far


def bfs_search(start, end):
    queue = Queue()
    queue.put(start)
    came_from = {start: None}

    while not queue.empty():
        current = queue.get()

        if current == end:
            break

        for next in get_neighbors(current, diagonal=False):  # Removed graph parameter
            if next not in came_from:
                queue.put(next)
                came_from[next] = current
    return came_from


def dijkstra(start, end):
    queue = [(0, start)]
    distances = {start: 0}
    previous_nodes = {start: None}
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node == end:
            break
        for neighbor in get_neighbors(current_node):
            neighbor_distance = current_distance + grid[neighbor[0]][neighbor[1]]
            if neighbor_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = neighbor_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (neighbor_distance, neighbor))
    return previous_nodes, distances

def kruskal_mst(vertices, edges):
    def find_parent(parent, i):
        if parent[i] == i:
            return i
        return find_parent(parent, parent[i])

    def union(parent, rank, x, y):
        root_x = find_parent(parent, x)
        root_y = find_parent(parent, y)

        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

    edges = sorted(edges)
    parent = [i for i in range(vertices)]
    rank = [0] * vertices
    result = []

    for edge in edges:
        weight, u, v = edge

        root_u = find_parent(parent, u)
        root_v = find_parent(parent, v)

        if root_u != root_v:
            result.append((u, v, weight))
            union(parent, rank, root_u, root_v)

    return result


def mst_search():
    global start, end
    edges = []
    for i in range(ROWS):
        for j in range(COLS):
            if i < ROWS - 1:
                edges.append((abs(grid[i][j] - grid[i + 1][j]), (i, j), (i + 1, j)))
            if j < COLS - 1:
                edges.append((abs(grid[i][j] - grid[i][j + 1]), (i, j), (i, j + 1)))

    mst_edges = kruskal_mst(ROWS * COLS, edges)

    for edge in mst_edges:
        i, j = edge[0]
        next_i, next_j = edge[1]
        canvas.create_line(
            j * CELL_SIZE + CELL_SIZE // 2,
            i * CELL_SIZE + CELL_SIZE // 2,
            next_j * CELL_SIZE + CELL_SIZE // 2,
            next_i * CELL_SIZE + CELL_SIZE // 2,
            fill="purple",
            width=2,
        )
# Path reconstruction with added error handling
def reconstruct_path(came_from, start, end):
    current = end
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)  # Use get to avoid KeyError
        if current is None:  # If part of the path is missing, return an empty path
            return []
    path.append(start)
    path.reverse()
    return path


def draw_grid():
    for i in range(ROWS):
        for j in range(COLS):
            color = OPEN_COLOR
            if grid[i][j] == 0:  # A wall
                color = WALL_COLOR
            canvas.create_rectangle(
                j * CELL_SIZE,
                i * CELL_SIZE,
                (j + 1) * CELL_SIZE,
                (i + 1) * CELL_SIZE,
                fill=color,
                outline="gray",
            )


def visualize_path(path):
    for position in path:
        i, j = position
        canvas.create_rectangle(
            j * CELL_SIZE,
            i * CELL_SIZE,
            (j + 1) * CELL_SIZE,
            (i + 1) * CELL_SIZE,
            fill=PATH_COLOR,
            outline="",
        )

    # Draw start and end on top
    if start:
        i, j = start
        canvas.create_rectangle(
            j * CELL_SIZE,
            i * CELL_SIZE,
            (j + 1) * CELL_SIZE,
            (i + 1) * CELL_SIZE,
            fill=START_COLOR,
            outline="",
        )
    if end:
        i, j = end
        canvas.create_rectangle(
            j * CELL_SIZE,
            i * CELL_SIZE,
            (j + 1) * CELL_SIZE,
            (i + 1) * CELL_SIZE,
            fill=END_COLOR,
            outline="",
        )


def handle_cell_click(event):
    global start, end
    x, y = event.x // CELL_SIZE, event.y // CELL_SIZE

    if not start:  # Set the start point
        start = (y, x)
        grid[y][x] = 1  # Make sure the start is not a wall
    elif not end and (y, x) != start:  # Set the end point
        end = (y, x)
        grid[y][x] = 1  # Make sure the end is not a wall
    else:  # Set or remove walls and weights
        if grid[y][x] != 0:
            grid[y][x] = 0  # Set wall
        else:
            grid[y][x] = 1  # Remove wall
    draw_grid()  # Redraw the grid to show the update


def handle_cell_click(event):
    global start, end
    x, y = event.x // CELL_SIZE, event.y // CELL_SIZE

    if not start:  # Set the start point
        start = (y, x)
        grid[y][x] = 1  # Make sure the start is not a wall
    elif not end and (y, x) != start:  # Set the end point
        end = (y, x)
        grid[y][x] = 1  # Make sure the end is not a wall
    else:  # Set or remove walls and weights
        if grid[y][x] != 0:
            grid[y][x] = 0  # Set wall
        else:
            grid[y][x] = 1  # Remove wall
    draw_grid()  # Redraw the grid to show the update


def start_search():
    algorithm = algorithm_choice.get()
    if algorithm == "A*":
        came_from, _ = a_star_search(start, end)
    elif algorithm == "BFS":
        came_from = bfs_search(start, end)
    else:  # Dijkstra's algorithm as default
        came_from, _ = dijkstra(start, end)  # Corrected function name

    path = reconstruct_path(came_from, start, end)
    if path:  # Only visualize if a path was found
        visualize_path(path)
    else:
        print("No path found!")  # Or handle this case in your GUI


# Set up the main window
root = tk.Tk()
root.title("Pathfinding Visualizer")

# Set up the canvas
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# Set up algorithm selection
algorithm_choice = tk.StringVar(root)
algorithm_choice.set("Dijkstra")  # Default value

algorithm_options = ["Dijkstra", "A*", "BFS", "MST"]  # Include "MST" in the options
algorithm_menu = tk.OptionMenu(root, algorithm_choice, *algorithm_options)
algorithm_menu.pack()

# Bind the canvas click event
canvas.bind("<Button-1>", handle_cell_click)

# Start search button
search_button = tk.Button(root, text="Start Search", command=start_search)
search_button.pack()

# Draw the initial grid
draw_grid()


root.mainloop()