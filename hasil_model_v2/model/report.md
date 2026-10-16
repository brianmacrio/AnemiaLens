# Laporan validasi AnemiaLens (out-of-fold, split per pasien)

Data: 624 foto, 624 pasien, prevalensi anemia 43.1%.

Anggota ensemble: color, color_rf, cnn (dipilih dari AUC out-of-fold; model lain hanya dilaporkan).

AUC ensemble: **0.841** (95% CI bootstrap per pasien 0.810–0.870).

Ambang skrining (sensitivitas ≥ 85%): 0.426; ambang risiko tinggi (spesifisitas ≥ 90%): 0.614.


## Per model

| model    |       n |   anemia |   auc |   sens |   spec |   ppv |   npv |   bal_acc |   hb_mae |   hb_r |   hb_within1 |   hb_within2 |
|:---------|--------:|---------:|------:|-------:|-------:|------:|------:|----------:|---------:|-------:|-------------:|-------------:|
| color    | 624.000 |  269.000 | 0.757 |  0.851 |  0.501 | 0.564 | 0.817 |     0.676 |    1.532 |  0.559 |        0.407 |        0.713 |
| color_rf | 624.000 |  269.000 | 0.831 |  0.851 |  0.665 | 0.658 | 0.855 |     0.758 |    1.409 |  0.638 |        0.446 |        0.732 |
| dino     | 624.000 |  269.000 | 0.738 |  0.851 |  0.454 | 0.541 | 0.801 |     0.652 |    1.588 |  0.513 |        0.399 |        0.689 |
| cnn      | 624.000 |  269.000 | 0.818 |  0.855 |  0.580 | 0.607 | 0.841 |     0.718 |    1.484 |  0.581 |        0.436 |        0.726 |
| ensemble | 624.000 |  269.000 | 0.841 |  0.859 |  0.662 | 0.658 | 0.861 |     0.760 |    1.409 |  0.646 |        0.450 |        0.742 |


## Ensemble per sumber data

| source                 |       n |   anemia |   auc |   sens |   spec |   ppv |   npv |   bal_acc |   hb_mae |   hb_r |   hb_within1 |   hb_within2 |
|:-----------------------|--------:|---------:|------:|-------:|-------:|------:|------:|----------:|---------:|-------:|-------------:|-------------:|
| cp_anemic              | 407.000 |  178.000 | 0.794 |  0.837 |  0.603 | 0.621 | 0.826 |     0.720 |    1.470 |  0.410 |        0.455 |        0.713 |
| eyes_defy_anemia/India |  95.000 |   68.000 | 0.714 |  0.956 |  0.185 | 0.747 | 0.625 |     0.571 |    1.397 |  0.535 |        0.432 |        0.747 |
| eyes_defy_anemia/Italy | 122.000 |   23.000 | 0.925 |  0.739 |  0.929 | 0.708 | 0.939 |     0.834 |    1.216 |  0.716 |        0.451 |        0.836 |


## Leave-one-source-out (proksi validasi eksternal)

| source_uji             |   n |   auc_color |   auc_color_rf |   auc_dino |   auc_rata2 |
|:-----------------------|----:|------------:|---------------:|-----------:|------------:|
| cp_anemic              | 407 |       0.640 |          0.601 |      0.581 |       0.627 |
| eyes_defy_anemia/India |  95 |       0.702 |          0.714 |      0.715 |       0.750 |
| eyes_defy_anemia/Italy | 122 |       0.885 |          0.800 |      0.838 |       0.888 |