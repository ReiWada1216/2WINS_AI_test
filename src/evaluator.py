import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple

import src.config as config
from src.visualization import visualize_gradcam

def get_inference_transform() -> transforms.Compose:
    """
    スライディングウィンドウ推論用の変換。
    Cropは手動で行うため、ここではTensor化と正規化のみを行う。
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def extract_patches(image: Image.Image, patch_size: int = 224, stride: int = 224) -> List[Tuple[torch.Tensor, Tuple[int, int]]]:
    """
    画像からパッチを切り出す（スライディングウィンドウ）。
    
    Args:
        image (Image.Image): PIL画像
        patch_size (int): パッチサイズ
        stride (int): ストライド（移動幅）。デフォルトはパッチサイズと同じ（重複なし）。
    
    Returns:
        List[Tuple[torch.Tensor, Tuple[int, int]]]: (変換済みパッチテンソル, (x, y)座標) のリスト
    """
    w, h = image.size
    patches = []
    transform = get_inference_transform()
    
    # 簡易的なグリッド切り出し
    # 端が余る場合は無視するか、パディングする等の戦略があるが、
    # ここではシンプルにループ範囲内で切り出す
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            crop = image.crop((x, y, x + patch_size, y + patch_size))
            tensor = transform(crop)
            patches.append((tensor, (x, y)))
            
    return patches

def evaluate_test_set(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: str,
    threshold: float = config.THRESHOLD
) -> Dict[str, float]:
    """
    Test Setに対する最終評価を実行する。
    画像レベルでの判定（ORゲートロジック）を行う。
    
    Args:
        model (nn.Module): 学習済みモデル
        test_loader (DataLoader): テストデータローダー (Datasetからパスを取得するために使用)
        device (str): デバイス
        threshold (float): 不良品判定の閾値 (確率 > threshold ならBad)
        
    Returns:
        Dict[str, float]: 評価指標
    """
    model.eval()
    
    # 画像パスと正解ラベルの取得
    image_paths = test_loader.dataset.image_paths
    true_labels = test_loader.dataset.labels
    
    pred_labels = []
    
    # 可視化候補の保存用
    fp_instances = [] # False Positives (良品なのに不良と判定)
    tp_instances = [] # True Positives (不良品を正しく判定)
    
    print(f"テストセットの {len(image_paths)} 枚の画像を評価中...")
    
    with torch.no_grad():
        for i, (img_path, label) in enumerate(tqdm(zip(image_paths, true_labels), total=len(image_paths))):
            try:
                # 画像読み込み
                original_image = Image.open(img_path).convert("RGB")
                np_original = np.array(original_image)
                
                # パッチ抽出
                patches_data = extract_patches(original_image, patch_size=config.PATCH_SIZE, stride=config.PATCH_SIZE // 2) # オーバーラップありで少し密に
                
                if not patches_data:
                    # パッチが取れない場合（画像が小さい等）はGood(0)とみなすか、エラーにする
                    pred_labels.append(0)
                    continue
                
                # パッチをバッチ化して推論
                patch_tensors = torch.stack([p[0] for p in patches_data]).to(device)
                
                outputs = model(patch_tensors)
                probs = torch.softmax(outputs, dim=1) # (NumPatches, 2)
                
                # Badクラス(1)の確率
                bad_probs = probs[:, 1]
                
                # 最大スコアを持つパッチを探す
                max_score, max_idx = torch.max(bad_probs, 0)
                max_score = max_score.item()
                
                # 画像レベルの判定
                is_bad = 1 if max_score > threshold else 0
                pred_labels.append(is_bad)
                
                # 可視化用データの保存
                # 不良品判定された場合、その根拠となったパッチを保存対象にする
                best_patch_tensor = patch_tensors[max_idx].unsqueeze(0) # (1, C, H, W)
                best_patch_coords = patches_data[max_idx][1]
                
                # False Positive (良品(0)なのに不良(1)と判定)
                if label == 0 and is_bad == 1:
                    # パッチ画像を切り出し
                    x, y = best_patch_coords
                    patch_img_np = np_original[y:y+config.PATCH_SIZE, x:x+config.PATCH_SIZE]
                    
                    fp_instances.append({
                        "path": img_path,
                        "patch_tensor": best_patch_tensor,
                        "patch_image": patch_img_np,
                        "score": max_score
                    })
                
                # True Positive (不良品(1)を正しく不良(1)と判定)
                if label == 1 and is_bad == 1:
                    x, y = best_patch_coords
                    patch_img_np = np_original[y:y+config.PATCH_SIZE, x:x+config.PATCH_SIZE]
                    
                    tp_instances.append({
                        "path": img_path,
                        "patch_tensor": best_patch_tensor,
                        "patch_image": patch_img_np,
                        "score": max_score
                    })
                    
            except Exception as e:
                print(f"{img_path} の処理中にエラーが発生しました: {e}")
                pred_labels.append(0) # エラー時はとりあえずGood扱い
                
    # -----------------------------------------------------
    # 指標計算
    # -----------------------------------------------------
    acc = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    precision = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)
    
    print("\n" + "="*50)
    print("テストセット評価結果")
    print("="*50)
    print(f"正解率 (Accuracy) : {acc:.4f}")
    print(f"再現率 (Recall)   : {recall:.4f}")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"F1スコア          : {f1:.4f}")
    print("-" * 20)
    print("混同行列 (Confusion Matrix):")
    print(cm)
    print("-" * 20)
    
    # -----------------------------------------------------
    # 可視化実行
    # -----------------------------------------------------
    print("\nGrad-CAMによる可視化画像を生成中...")
    save_dir = config.RESULTS_DIR / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # True Positivesからスコアが高い上位3件
    tp_instances.sort(key=lambda x: x['score'], reverse=True)
    for i, item in enumerate(tp_instances[:3]):
        save_path = save_dir / f"TP_sample_{i}_{Path(item['path']).name}"
        visualize_gradcam(
            model=model,
            image_tensor=item['patch_tensor'],
            original_image=item['patch_image'],
            save_path=save_path,
            device=device,
            target_class=1
        )
        
    # False Positivesからスコアが高い上位3件
    fp_instances.sort(key=lambda x: x['score'], reverse=True)
    for i, item in enumerate(fp_instances[:3]):
        save_path = save_dir / f"FP_sample_{i}_{Path(item['path']).name}"
        visualize_gradcam(
            model=model,
            image_tensor=item['patch_tensor'],
            original_image=item['patch_image'],
            save_path=save_path,
            device=device,
            target_class=1 # なぜ不良と判定されたか知りたいのでBadクラスの活性化を見る
        )
    
    print(f"可視化結果を {save_dir} に保存しました")
    
    return {
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1
    }

if __name__ == "__main__":
    pass
