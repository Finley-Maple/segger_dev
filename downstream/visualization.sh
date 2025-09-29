# colon gene level static
cd /home/f833u/segger/downstream && python post_embedding_visualization.py --dataset colon --model_type seq --all_regions --align_loss --create_gene_level_plot 


# colon transcript level interactive
cd /home/f833u/segger/downstream && python post_embedding_visualization.py --dataset colon --model_type seq --spatial_region 2300 2500 2100 2300 --align_loss --create_interactive_plots

cd /home/f833u/segger/downstream && python post_embedding_visualization.py --dataset colon --model_type seq --spatial_region 2300 2500 2100 2300 --create_interactive_plots

cd /home/f833u/segger/downstream && python post_embedding_visualization.py --dataset breast --model_type seq --spatial_region 5000 5200 5800 6000 --align_loss --create_interactive_plots

cd /home/f833u/segger/downstream && python post_embedding_visualization.py --dataset breast --model_type seq --spatial_region 5000 5200 5800 6000 --create_interactive_plots