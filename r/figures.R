# Oncoscope result figures (ggplot2 + patchwork + ggridges, rendered with ragg).
# Usage: Rscript r/figures.R [run_dir] [out_dir]

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(ggridges); library(scales); library(jsonlite); library(forcats)
})

args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args) >= 1) args[[1]] else "runs/unet-resnet18"
out <- if (length(args) >= 2) args[[2]] else "figures"
ev  <- file.path(run, "eval")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

classes <- c("meningioma", "glioma", "pituitary")
pal <- c(meningioma = "#4F9DDE", glioma = "#EF6F6C", pituitary = "#F6C85F")
bg <- "#0E1117"; fg <- "#E8EAF0"; muted <- "#8B93A7"; grid <- "#222838"

theme_onc <- function(base = 13) {
  theme_minimal(base_size = base) +
    theme(plot.background = element_rect(fill = bg, colour = NA),
          panel.background = element_rect(fill = bg, colour = NA),
          panel.grid.major = element_line(colour = grid, linewidth = 0.3), panel.grid.minor = element_blank(),
          text = element_text(colour = fg), axis.text = element_text(colour = muted),
          plot.title = element_text(face = "bold", size = rel(1.3)),
          plot.subtitle = element_text(colour = muted, margin = margin(b = 10)),
          legend.position = "top", legend.justification = "left",
          strip.text = element_text(colour = fg, face = "bold"), plot.margin = margin(16, 18, 12, 14))
}
save <- function(p, name, w = 11, h = 7) {
  ggsave(file.path(out, name), p, width = w, height = h, dpi = 170, device = ragg::agg_png, bg = bg)
  message("wrote ", file.path(out, name))
}

m  <- fromJSON(file.path(ev, "metrics.json"))
pi <- read_csv(file.path(ev, "per_image.csv"), show_col_types = FALSE) |>
  mutate(true = factor(true, levels = classes), correct = true == pred)

# 1. Shortcut test - the headline figure -------------------------------------------------------------
sc <- read_csv(file.path(ev, "shortcut.csv"), show_col_types = FALSE) |>
  mutate(condition = factor(condition, levels = rev(condition)),
         kind = ifelse(grepl("baseline", condition), "baseline", "model"))
p_sc <- ggplot(sc, aes(accuracy, condition, fill = condition)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = percent(accuracy, 0.1)), hjust = -0.15, colour = fg, size = 4.4, fontface = "bold") +
  geom_vline(xintercept = 1 / 3, linetype = "dotted", colour = muted) +
  annotate("text", x = 1 / 3, y = 4.5, label = "random guess", colour = muted, size = 3.2, hjust = -0.05) +
  scale_fill_manual(values = c("#9DD9D2", "#EF6F6C", "#F6C85F", "#3a4256") |> setNames(rev(levels(sc$condition))),
                    guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0, 1.08), expand = c(0, 0)) +
  labs(x = "tumour-type accuracy on held-out patients", y = NULL,
       title = "Is it reading the tumour, or the rest of the scan?",
       subtitle = paste0("Same model, same test slices. Erasing the tumour should drop accuracy toward chance if the\n",
                         "classifier truly depends on the lesion; the gap shows how much it relies on anatomical context.")) +
  theme_onc() + theme(panel.grid.major.y = element_blank())
save(p_sc, "shortcut_test.png", w = 11, h = 5.2)

# 2. Confusion matrix + Dice by tumour type ---------------------------------------------------------
cm <- read_csv(file.path(ev, "confusion.csv"), show_col_types = FALSE) |>
  group_by(true) |> mutate(recall = n / sum(n)) |> ungroup() |>
  mutate(true = factor(true, levels = rev(classes)), pred = factor(pred, levels = classes))
p_cm <- ggplot(cm, aes(pred, true, fill = recall)) +
  geom_tile(colour = bg, linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.0f%%\n%d", 100 * recall, n), colour = recall > 0.55), size = 4, lineheight = 0.9) +
  scale_fill_viridis_c(option = "mako", limits = c(0, 1), labels = percent, name = "row %") +
  scale_colour_manual(values = c(`TRUE` = bg, `FALSE` = fg), guide = "none") +
  coord_equal() +
  labs(x = "predicted", y = "expert label", title = "Tumour type",
       subtitle = sprintf("slice accuracy %.1f%% · macro-F1 %.3f · patient-level accuracy %.1f%%",
                          100 * m$accuracy, m$macro_f1, 100 * m$patient_accuracy)) +
  theme_onc() + theme(panel.grid = element_blank(), legend.position = "right")

p_dice <- ggplot(pi, aes(dice, fct_rev(true), fill = true)) +
  geom_density_ridges(scale = 1.3, colour = bg, alpha = 0.9, quantile_lines = TRUE, quantiles = 2,
                      vline_colour = fg, rel_min_height = 0.005) +
  scale_fill_manual(values = pal, guide = "none") +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "Dice overlap with expert outline (line = median)", y = NULL, title = "Tumour outline",
       subtitle = sprintf("mean Dice %.3f · median %.3f · IoU %.3f", m$dice_mean, m$dice_median, m$iou_mean)) +
  theme_onc()
save((p_cm | p_dice) + plot_layout(widths = c(1, 1.15)), "classification_and_segmentation.png", w = 15, h = 6.2)

# 3. Where segmentation fails: small tumours ---------------------------------------------------------
p_area <- ggplot(pi, aes(tumour_area * 100, dice, colour = true)) +
  geom_point(alpha = 0.55, size = 1.6) +
  geom_smooth(se = FALSE, method = "loess", formula = y ~ x, span = 1, linewidth = 1.1) +
  scale_x_log10(labels = \(x) paste0(x, "%")) +
  scale_colour_manual(values = pal, name = NULL) +
  labs(x = "tumour size (% of image, log scale)", y = "Dice",
       title = "Glioma is the hard one, at every size",
       subtitle = paste0("Each point is one held-out slice. Meningioma and pituitary outlines sit near Dice 0.9 whatever their size;\n",
                         "gliomas infiltrate surrounding tissue with irregular borders, and some are missed entirely (Dice 0).")) +
  theme_onc()
save(p_area, "dice_vs_tumour_size.png", w = 10, h = 6.5)

# 4. Grad-CAM localisation + calibration ------------------------------------------------------------
pi <- pi |> mutate(focus = cam_in_tumour / pmax(tumour_area, 1e-6))
p_cam <- ggplot(pi, aes(tumour_area, focus, colour = correct)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = muted) +
  geom_point(alpha = 0.6, size = 1.6) +
  annotate("text", x = min(pi$tumour_area), y = 0.93, hjust = 0, vjust = 1, colour = muted, size = 3.3,
           label = "1x = attention spread evenly over the image") +
  scale_x_log10(labels = percent) +
  scale_y_log10(labels = \(x) paste0(x, "x")) +
  scale_colour_manual(values = c(`TRUE` = "#9DD9D2", `FALSE` = "#EF6F6C"),
                      labels = c(`TRUE` = "correct", `FALSE` = "wrong"), name = NULL) +
  labs(x = "tumour area (share of image, log)", y = "Grad-CAM attention density in tumour vs uniform",
       title = "Does the classifier look at the tumour?",
       subtitle = sprintf(paste0("Median focus %.1fx: attention is denser on the tumour than elsewhere, but diffuse; the attention\n",
                                 "peak lands on the tumour (+8 px) in only %.0f%% of slices. Consistent with the shortcut test."),
                          median(pi$focus), 100 * m$cam_pointing_game)) +
  theme_onc()
rel <- read_csv(file.path(ev, "reliability.csv"), show_col_types = FALSE) |> filter(n >= 5)
p_rel <- ggplot(rel, aes(confidence, accuracy)) +
  geom_abline(linetype = "dashed", colour = muted) +
  geom_line(colour = "#F6C85F", linewidth = 1) + geom_point(aes(size = n), colour = "#F6C85F") +
  scale_size_area(max_size = 7, guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0.3, 1)) + scale_y_continuous(labels = percent, limits = c(0, 1)) +
  labs(x = "confidence", y = "accuracy", title = "Calibration",
       subtitle = sprintf("ECE %.3f · %.0f%% of slices predicted at >90%% confidence\n(bins with fewer than 5 slices hidden)",
                          m$ece, 100 * mean(pmax(pi$p_meningioma, pi$p_glioma, pi$p_pituitary) > 0.9))) +
  theme_onc()
save((p_cam | p_rel) + plot_layout(widths = c(1.5, 1)), "gradcam_and_calibration.png", w = 15, h = 6.5)

# 5. Training curves --------------------------------------------------------------------------------
hist <- read_csv(file.path(run, "history.csv"), show_col_types = FALSE) |>
  select(epoch, `seg loss` = seg_loss, `cls loss` = cls_loss, `val Dice` = val_dice, `val macro-F1` = val_macro_f1) |>
  pivot_longer(-epoch)
p_hist <- ggplot(hist, aes(epoch, value, colour = name)) +
  geom_line(linewidth = 1) + geom_point(size = 1.4) +
  facet_wrap(~name, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = c("#EF6F6C", "#F6C85F", "#4F9DDE", "#9DD9D2"), guide = "none") +
  labs(x = "epoch", y = NULL, title = "Training (model selected on validation patients only)") + theme_onc()
save(p_hist, "training.png", w = 14, h = 4.2)
