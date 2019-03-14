#!/usr/bin/env bash
# Generating bipartite networks with 50 districts.
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 4.2 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 50 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.00102 20 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 50 #50
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 5 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 50 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.00099 42 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 50 #50
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 6.1 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 50 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.000959 26 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 50 #50
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 3.5 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 50 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.00092 14 #50
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 50 #50

## Generating bipartite networks with 100 districts.
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 2.65 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 100 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.00105 13 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100 #100
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 3.1 #99
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 100 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.001029 30 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100 #100
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 3.9 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 100 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.00098 21 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100 #100
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 2.11 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 100 #100
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 2 --args 0.00141 10 #97
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100 #100

# Generating bipartite networks with the best silhouette.
python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 2.866999999999975
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 75
#python generate_bipartite_network.py --input_file 'data/input/cities/Charlotte.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 89
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 3.51799999999999
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 92
#python generate_bipartite_network.py --input_file 'data/input/cities/Las Vegas.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 5.1759999999999895
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 94
#python generate_bipartite_network.py --input_file 'data/input/cities/Phoenix.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100
#
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 0 --args 'checkins' 2.262899999999998
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 1 --args 80
#python generate_bipartite_network.py --input_file 'data/input/cities/Pittsburgh.json' --output_dir 'data/output/' --clustering_algorithm 3 --args 100
