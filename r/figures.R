# Oncoscope figures in a journal / MATLAB-like style (white background, boxed axes, inward ticks, Arial,
# MATLAB default colour order, parula colormap). Captions live in the README, not inside the figures.
# Usage: Rscript r/figures.R [run_dir] [out_dir]

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(scales); library(jsonlite); library(forcats)
})

args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args) >= 1) args[[1]] else "runs/unet-resnet18"
out <- if (length(args) >= 2) args[[2]] else "figures"
ev  <- file.path(run, "eval")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

mlab   <- c(blue = "#0072BD", orange = "#D95319", yellow = "#EDB120", purple = "#7E2F8E",
            green = "#77AC30", cyan = "#4DBEEE", red = "#A2142F")
parula <- c("#352A87", "#0F5CDD", "#1481D6", "#06A4CA", "#2EB7A4", "#87BF77", "#D1BB59", "#FEC832", "#F9FB0E")
classes <- c("meningioma", "glioma", "pituitary")
labels_cap <- c(meningioma = "Meningioma", glioma = "Glioma", pituitary = "Pituitary")
pal <- c(meningioma = mlab[["blue"]], glioma = mlab[["orange"]], pituitary = mlab[["yellow"]])

theme_paper <- function(base = 10) {
  theme_classic(base_size = base, base_family = "Arial") +
    theme(
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
      axis.line = element_blank(),
      axis.ticks = element_line(colour = "black", linewidth = 0.4),
      axis.ticks.length = unit(-3, "pt"),
      axis.text = element_text(colour = "black"),
      axis.text.x = element_text(margin = margin(t = 5)),
      axis.text.y = element_text(margin = margin(r = 5)),
      axis.title = element_text(colour = "black"),
      plot.title = element_text(face = "plain", hjust = 0.5, size = rel(1.05)),
      plot.tag = element_text(face = "bold", size = rel(1.25)),
      legend.background = element_rect(colour = "black", linewidth = 0.3, fill = "white"),
      legend.key = element_blank(), legend.title = element_blank(),
      strip.background = element_blank(), strip.text = element_text(colour = "black"),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(8, 10, 8, 8)
    )
}
save <- function(p, name, w, h) {
  ggsave(file.path(out, name), p, width = w, height = h, dpi = 300, device = ragg::agg_png, bg = "white")
  message("wrote ", file.path(out, name))
}
tags <- plot_annotation(tag_levels = "a")
inside_br <- theme(legend.position = "inside", legend.position.inside = c(0.97, 0.03), legend.justification = c(1, 0))

m  <- fromJSON(file.path(ev, "metrics.json"))
pi <- read_csv(file.path(ev, "per_image.csv"), show_col_types = FALSE) |>
  mutate(true = factor(true, levels = classes), correct = true == pred,
         focus = cam_in_tumour / pmax(tumour_area, 1e-6))

# 1. Shortcut test: overall and by tumour type -----------------------------------------------------------
sc <- read_csv(file.path(ev, "shortcut.csv"), show_col_types = FALSE) |>
  mutate(condition = recode(condition, `full image` = "Full image", `tumour only (context blanked)` = "Tumour only",
                            `tumour erased (context only)` = "Tumour erased", `majority-class baseline` = "Majority class"),
         condition = factor(condition, levels = c("Full image", "Tumour erased", "Tumour only", "Majority class")))
p_sc <- ggplot(sc, aes(condition, accuracy, fill = condition == "Majority class")) +
  geom_col(width = 0.6, colour = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.1f", 100 * accuracy)), vjust = -0.5, size = 3.2, family = "Arial") +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  scale_fill_manual(values = c(`FALSE` = mlab[["blue"]], `TRUE` = "grey75"), guide = "none") +
  scale_y_continuous(labels = \(x) x * 100, limits = c(0, 1.05), expand = c(0, 0)) +
  labs(x = NULL, y = "Accuracy (%)") + theme_paper()

by_class <- pi |> group_by(true) |>
  summarise(`Full image` = mean(pred == true), `Tumour erased` = mean(pred_tumour_erased == true), .groups = "drop") |>
  pivot_longer(-true, names_to = "condition", values_to = "accuracy") |>
  mutate(true = factor(labels_cap[as.character(true)], levels = labels_cap))
p_cls <- ggplot(by_class, aes(true, accuracy, fill = condition)) +
  geom_col(position = position_dodge(0.7), width = 0.65, colour = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.1f", 100 * accuracy)), position = position_dodge(0.7), vjust = -0.5,
            size = 3, family = "Arial") +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  scale_fill_manual(values = c(`Full image` = mlab[["blue"]], `Tumour erased` = mlab[["orange"]])) +
  scale_y_continuous(labels = \(x) x * 100, limits = c(0, 1.1), expand = c(0, 0)) +
  labs(x = "True tumour type", y = "Accuracy (%)") +
  theme_paper() + theme(legend.position = "top", legend.justification = "right",
                        legend.background = element_blank(), legend.margin = margin(0, 0, -4, 0),
                        legend.key.size = unit(9, "pt"))
save((p_sc | p_cls) + plot_layout(widths = c(1, 1.15)) + tags, "shortcut_test.png", w = 9, h = 3.6)

# 2. Confusion matrix + Dice by tumour type ------------------------------------------------------------
cm <- read_csv(file.path(ev, "confusion.csv"), show_col_types = FALSE) |>
  group_by(true) |> mutate(recall = n / sum(n)) |> ungroup() |>
  mutate(true = factor(labels_cap[true], levels = rev(labels_cap)), pred = factor(labels_cap[pred], levels = labels_cap))
p_cm <- ggplot(cm, aes(pred, true, fill = recall)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f\n(%d)", recall, n), colour = recall > 0.55), size = 3, lineheight = 0.9,
            family = "Arial") +
  scale_fill_gradientn(colours = parula, limits = c(0, 1), name = "Recall") +
  scale_colour_manual(values = c(`TRUE` = "black", `FALSE` = "white"), guide = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) +
  coord_equal() +
  labs(x = "Predicted type", y = "True type") +
  theme_paper() + theme(legend.position = "right", legend.background = element_blank(),
                        legend.title = element_text(size = rel(0.85)), axis.ticks = element_blank())
set.seed(1)
p_dice <- ggplot(pi |> mutate(lab = factor(labels_cap[as.character(true)], levels = labels_cap)),
                 aes(lab, dice, colour = true)) +
  geom_jitter(width = 0.18, size = 0.7, alpha = 0.5) +
  geom_boxplot(fill = NA, colour = "black", outlier.shape = NA, width = 0.5, linewidth = 0.4) +
  scale_colour_manual(values = pal, guide = "none") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "True tumour type", y = "Dice coefficient") + theme_paper()
save((p_cm | p_dice) + plot_layout(widths = c(1.15, 1)) + tags, "classification_and_segmentation.png", w = 9, h = 3.8)

# 3. Dice vs tumour size ---------------------------------------------------------------------------------
p_area <- ggplot(pi, aes(tumour_area * 100, dice, colour = true, shape = true)) +
  geom_point(size = 1.1, alpha = 0.6) +
  geom_smooth(aes(linetype = true), se = FALSE, method = "loess", formula = y ~ x, span = 1, linewidth = 0.8) +
  scale_x_log10() +
  scale_colour_manual(values = pal, labels = labels_cap) +
  scale_shape_manual(values = c(16, 17, 15), labels = labels_cap) +
  scale_linetype_manual(values = c("solid", "solid", "solid"), labels = labels_cap) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Tumour area (% of image, log scale)", y = "Dice coefficient") +
  theme_paper() + inside_br
save(p_area, "dice_vs_tumour_size.png", w = 5.2, h = 4)

# 4. Grad-CAM focus + calibration ------------------------------------------------------------------------
p_cam <- ggplot(pi, aes(tumour_area * 100, focus, colour = correct, shape = correct)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  geom_point(size = 1.2, alpha = 0.7) +
  scale_x_log10() + scale_y_log10() +
  scale_colour_manual(values = c(`TRUE` = mlab[["blue"]], `FALSE` = mlab[["orange"]]),
                      labels = c(`TRUE` = "Correct", `FALSE` = "Misclassified")) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 4), labels = c(`TRUE` = "Correct", `FALSE` = "Misclassified")) +
  labs(x = "Tumour area (% of image, log scale)", y = "Grad-CAM focus on tumour (x uniform)") +
  theme_paper() + inside_br
rel <- read_csv(file.path(ev, "reliability.csv"), show_col_types = FALSE) |> filter(n >= 5)
p_rel <- ggplot(rel, aes(confidence, accuracy)) +
  geom_abline(linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  geom_line(colour = mlab[["blue"]], linewidth = 0.6) + geom_point(colour = mlab[["blue"]], size = 1.8) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) + scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  coord_equal() +
  labs(x = "Confidence", y = "Accuracy") + theme_paper()
save((p_cam | p_rel) + plot_layout(widths = c(1.4, 1)) + tags, "gradcam_and_calibration.png", w = 9, h = 3.9)

# 5. Training curves -------------------------------------------------------------------------------------
hist <- read_csv(file.path(run, "history.csv"), show_col_types = FALSE)
loss <- hist |> select(epoch, Segmentation = seg_loss, Classification = cls_loss) |> pivot_longer(-epoch)
p_loss <- ggplot(loss, aes(epoch, value, colour = name, shape = name)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  scale_colour_manual(values = c(Segmentation = mlab[["blue"]], Classification = mlab[["orange"]])) +
  scale_shape_manual(values = c(Segmentation = 16, Classification = 17)) +
  labs(x = "Epoch", y = "Training loss") + theme_paper() +
  theme(legend.position = "inside", legend.position.inside = c(0.97, 0.97), legend.justification = c(1, 1))
val <- hist |> select(epoch, Dice = val_dice, `Macro-F1` = val_macro_f1) |> pivot_longer(-epoch)
p_val <- ggplot(val, aes(epoch, value, colour = name, shape = name)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  scale_colour_manual(values = c(Dice = mlab[["purple"]], `Macro-F1` = mlab[["green"]])) +
  scale_shape_manual(values = c(Dice = 15, `Macro-F1` = 16)) +
  labs(x = "Epoch", y = "Validation score") + theme_paper() + inside_br
save((p_loss | p_val) + tags, "training.png", w = 9, h = 3.2)
