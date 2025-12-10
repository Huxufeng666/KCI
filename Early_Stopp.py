# ==================================================
# 5.1 早停机制类
# ==================================================
class EarlyStopping:
    """
    早停机制：当验证损失不再下降时，停止训练
    """
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        """
        Args:
            patience (int): 验证损失未改善时，等待的轮数
            min_delta (float): 最小改善阈值
            restore_best_weights (bool): 是否恢复最优权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def step(self, val_loss, model,epoch):
        """
        每轮验证后调用
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            print(f"No improvement. Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"Early stopping triggered after epoch {epoch}")
                self.early_stop = True
                if self.restore_best_weights:
                    print("Restoring best weights...")
                    model.load_state_dict(self.best_weights)