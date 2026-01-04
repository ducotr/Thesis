# size_table = {}
# for method in NodeScaling:
#     G = build_cooccurrence_network(
#         publications=udt_pc,
#         min_value=20,
#         node_scaling=method,
#         edge_scaling=EdgeScaling.NONE
#     )
#     for node, size in G.nodes(data="value"):
#         if node not in size_table:
#             size_table[node] = {}
#         size_table[node][method.name] = size
# methods = [m.name for m in NodeScaling]
# header = ["Node"] + methods
# print(f"{header[0]:<27}", end="")
# for m in methods:
#     print(f"{m:<20}", end="")
# print()
# for node, method_sizes in size_table.items():
#     print(f"{node:<27}", end="")
#     for m in methods:
#         val = method_sizes.get(m, "-")
#         print(f"{str(val):<20}", end="")
#     print()

# print("\n\n\n")

# width_table = {}
# for method in EdgeScaling:
#     G = build_cooccurrence_network(
#         publications=udt_pc,
#         min_value=20,
#         node_scaling=NodeScaling.NONE,
#         edge_scaling=method
#     )
#     for u, v, width in G.edges(data="weight"):
#         if (u, v) not in width_table:
#             width_table[(u, v)] = {}
#         width_table[(u, v)][method.name] = width
# methods = [m.name for m in EdgeScaling]
# header = ["Edge"] + methods
# print(f"{header[0]:<51}", end="")
# for m in methods:
#     print(f"{m:<20}", end="")
# print()
# for edge, method_width in width_table.items():
#     print(f"{str(edge):<51}", end="")
#     for m in methods:
#         val = method_width.get(m, "-")
#         print(f"{str(val):<20}", end="")
#     print()
