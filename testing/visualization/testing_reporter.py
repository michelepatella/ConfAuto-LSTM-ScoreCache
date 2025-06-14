from utils.logs.log_utils import info


def generate_model_evaluation_report(
        metrics,
        top_k_accuracy,
        kappa_statistic,
        avg_loss,
        config_settings
):
    """
    Method to show an evaluation report.
    :param metrics: The class report metrics.
    :param top_k_accuracy: The top-k accuracy.
    :param kappa_statistic: The kappa statistic value.
    :param avg_loss: The average loss value.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Evaluation report printing started...")

    try:
        # title
        print("\n" + "=" * 85)
        print(" " * 30 + "Model Standalone Evaluation Report")
        print("=" * 85 + "\n")

        # average loss, top-k accuracy, and kappa statistic
        print(f"Average Loss:                           {avg_loss:.4f}\n")
        print(f"Top-{config_settings.top_k} Accuracy:                         {top_k_accuracy:.4f}\n")
        print(f"Kappa Statistic:                        {kappa_statistic:.4f}\n")

        # class report per class
        print("Class Report per Class:\n")
        header = f"{'Class':<6} | {'Precision':>9} | {'Recall':>7} | {'F1-Score':>8} | {'Support':>7}"
        print(header)
        print("-" * len(header))
        for cls in (k for k in metrics.keys() if k.isdigit()):
            v = metrics[cls]
            print(f"{cls:<6} | {v['precision']:9.4f} | {v['recall']:7.4f} | {v['f1-score']:8.4f} | {int(v['support']):7}")

        # summary of metrics
        print("\nSummary:\n")
        # accuracy
        print(f"Accuracy:                     {metrics.get('accuracy', 0):.4f}")
        # macro avg
        macro = metrics.get('macro avg', {})
        print(f"Macro Avg Precision:          {macro.get('precision', 0):.4f}")
        print(f"Macro Avg Recall:             {macro.get('recall', 0):.4f}")
        print(f"Macro Avg F1-Score:           {macro.get('f1-score', 0):.4f}")
        # weighted avg
        weighted = metrics.get('weighted avg', {})
        print(f"Weighted Avg Precision:       {weighted.get('precision', 0):.4f}")
        print(f"Weighted Avg Recall:          {weighted.get('recall', 0):.4f}")
        print(f"Weighted Avg F1-Score:        {weighted.get('f1-score', 0):.4f}")

        print("\n" + "=" * 85 + "\n")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # print a successful message
    info("🟢 Evaluation report printed.")