//! C exports for `ferro_ta_core::price_transform`.

crate::ffi_exports! {
    group: "price_transform",
    /// Average Price: (open + high + low + close) / 4.
    ft_avgprice = price_transform::avgprice(open, high, low, close)[]
        -> [out: f64];
    /// Median Price: (high + low) / 2.
    ft_medprice = price_transform::medprice(high, low)[]
        -> [out: f64];
    /// Typical Price: (high + low + close) / 3.
    ft_typprice = price_transform::typprice(high, low, close)[]
        -> [out: f64];
    /// Weighted Close Price: (high + low + close * 2) / 4.
    ft_wclprice = price_transform::wclprice(high, low, close)[]
        -> [out: f64];
}
