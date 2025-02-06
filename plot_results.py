import matplotlib.pyplot as plt
from numpy import log10

def plot_rank_metric_both(test_results, birank_test_result, file_name):
    top_values = list(test_results.keys())
    hr_list = [values['HR@N'] for values in test_results.values()]
    ndcg_list = [values['NDCG@N'].item() for values in test_results.values()]
    mrr_list = [values['MRR'].item() for values in test_results.values()]

    hr_list_birank = [values['HR@N'] for values in birank_test_result.values()]
    ndcg_list_birank = [values['NDCG@N'].item() for values in birank_test_result.values()]
    mrr_list_birank = [values['MRR'].item() for values in birank_test_result.values()]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    
    # Plot HR@N
    axs[0].plot(top_values, hr_list, marker='o', color='b', label="Matrix Factorization")
    axs[0].plot(top_values, hr_list_birank, marker='o', color='g', label="Birank")
    axs[0].set_title('Hit Rate (HR@K)')
    axs[0].set_xlabel('Top-K')
    axs[0].set_ylabel('HR@K')
    axs[0].grid(True)
    axs[0].legend()

    # Plot NDCG@N
    axs[1].plot(top_values, ndcg_list, marker='s', color='b', label="Matrix Factorization")
    axs[1].plot(top_values, ndcg_list_birank, marker='s', color='g', label="Birank")
    axs[1].set_title('Normalized Discounted Cumulative Gain (NDCG@K)')
    axs[1].set_xlabel('Top-K')
    axs[1].set_ylabel('NDCG@K')
    axs[1].grid(True)
    axs[1].legend()

    # Plot MRR
    axs[2].plot(top_values, mrr_list, marker='^', color='b', label="Matrix Factorization")
    axs[2].plot(top_values, mrr_list_birank, marker='^', color='g', label="Birank")
    axs[2].set_title('Mean Reciprocal Rank (MRR)')
    axs[2].set_xlabel('Top-K')
    axs[2].set_ylabel('MRR')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"{file_name}.eps")
    plt.show()

def plot_rank_metric(test_results, file_name):
    top_values = list(test_results.keys())
    hr_list = [values['HR@N'] for values in test_results.values()]
    ndcg_list = [values['NDCG@N'].item() for values in test_results.values()]
    mrr_list = [values['MRR'].item() for values in test_results.values()]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    
    # Plot HR@N
    axs[0].plot(top_values, hr_list, marker='o', color='b', label="HR@K")
    axs[0].set_title('Hit Rate (HR@K)')
    axs[0].set_xlabel('Top-K')
    axs[0].set_ylabel('HR@K')
    axs[0].grid(True)
    axs[0].legend()

    # Plot NDCG@N
    axs[1].plot(top_values, ndcg_list, marker='s', color='g', label="NDCG@K")
    axs[1].set_title('Normalized Discounted Cumulative Gain (NDCG@K)')
    axs[1].set_xlabel('Top-K')
    axs[1].set_ylabel('NDCG@K')
    axs[1].grid(True)
    axs[1].legend()

    # Plot MRR
    axs[2].plot(top_values, mrr_list, marker='^', color='r', label="MRR")
    axs[2].set_title('Mean Reciprocal Rank (MRR)')
    axs[2].set_xlabel('Top-K')
    axs[2].set_ylabel('MRR')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"{file_name}.eps")
    plt.show()

def plot_train_progress(history, title, file_name):
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  fig.suptitle(title, fontsize=14, fontweight='bold')
  axes[0].plot(history['loss'], label='loss')
  axes[0].set_title('Loss')
  axes[0].set_xlabel('Epochs')
  axes[0].set_ylabel('Loss')
  axes[0].grid(True)

  axes[1].plot(history['train_rmse'], label='Train RMSE')
  axes[1].plot(history['test_rmse'], label='Test RMSE')
  axes[1].set_title('Train and Test RMSE')
  axes[1].set_xlabel('Epochs')
  axes[1].set_ylabel('RMSE')
  axes[1].legend()
  axes[1].grid(True)

  plt.tight_layout()
  plt.savefig(f"{file_name}.eps")
  plt.show()