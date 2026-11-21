use brush_dataset::depth::turbo_u8;
use egui::{Align2, Color32, Rect};

/// Vertical turbo colorbar with the depth range labelled in the same units
/// as the map (metres when the depth maps are metric).
pub(crate) fn draw_depth_colorbar(ui: &egui::Ui, rect: Rect, depth_min: f32, depth_max: f32) {
    let painter = ui.painter_at(rect);
    let bar_w = 16.0;
    let bar_h = (rect.height() * 0.4).clamp(80.0, 240.0);
    let bar_rect = Rect::from_min_size(
        egui::pos2(rect.max.x - 24.0 - bar_w, rect.min.y + 36.0),
        egui::vec2(bar_w, bar_h),
    );
    let bg_rect = Rect::from_min_max(
        egui::pos2(bar_rect.min.x - 52.0, bar_rect.min.y - 12.0),
        egui::pos2(bar_rect.max.x + 12.0, bar_rect.max.y + 12.0),
    );
    painter.rect_filled(
        bg_rect,
        6.0,
        Color32::from_rgba_unmultiplied(20, 20, 25, 180),
    );

    let steps = 64;
    for i in 0..steps {
        let f = i as f32 / (steps - 1) as f32;
        let seg_top = bar_rect.max.y - (i as f32 + 1.0) / steps as f32 * bar_h;
        let rgb = turbo_u8(f);
        painter.rect_filled(
            Rect::from_min_size(
                egui::pos2(bar_rect.min.x, seg_top),
                egui::vec2(bar_w, bar_h / steps as f32 + 1.0),
            ),
            0.0,
            Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
        );
    }

    let font = egui::FontId::new(11.0, egui::FontFamily::Proportional);
    painter.text(
        egui::pos2(bar_rect.min.x - 6.0, bar_rect.min.y),
        Align2::RIGHT_CENTER,
        format!("{depth_max:.2}"),
        font.clone(),
        Color32::WHITE,
    );
    painter.text(
        egui::pos2(bar_rect.min.x - 6.0, bar_rect.max.y),
        Align2::RIGHT_CENTER,
        format!("{depth_min:.2}"),
        font,
        Color32::WHITE,
    );
}
