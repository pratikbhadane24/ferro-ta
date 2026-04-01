//! Currency metadata and Indian number formatting.

/// Immutable currency descriptor.
///
/// Carries the currency code, symbol, decimal places, and whether to use
/// Indian lakh/crore grouping (1,23,45,678.00) instead of standard
/// Western grouping (1,234,567.89).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Currency {
    /// IETF currency code, e.g. "INR", "USD".
    pub code: &'static str,
    /// Display symbol, e.g. "₹", "$".
    pub symbol: &'static str,
    /// Number of decimal places for formatting.
    pub decimal_places: u8,
    /// Use Indian lakh/crore digit grouping (true only for INR).
    pub lakh_grouping: bool,
}

impl Currency {
    pub const INR: Currency = Currency {
        code: "INR",
        symbol: "₹",
        decimal_places: 2,
        lakh_grouping: true,
    };
    pub const USD: Currency = Currency {
        code: "USD",
        symbol: "$",
        decimal_places: 2,
        lakh_grouping: false,
    };
    pub const EUR: Currency = Currency {
        code: "EUR",
        symbol: "€",
        decimal_places: 2,
        lakh_grouping: false,
    };
    pub const GBP: Currency = Currency {
        code: "GBP",
        symbol: "£",
        decimal_places: 2,
        lakh_grouping: false,
    };
    pub const JPY: Currency = Currency {
        code: "JPY",
        symbol: "¥",
        decimal_places: 0,
        lakh_grouping: false,
    };
    pub const USDT: Currency = Currency {
        code: "USDT",
        symbol: "₮",
        decimal_places: 2,
        lakh_grouping: false,
    };

    /// Look up a currency by IETF code (case-insensitive).
    /// Returns `None` if the code is not recognised.
    pub fn from_code(code: &str) -> Option<&'static Currency> {
        match code.to_ascii_uppercase().as_str() {
            "INR" => Some(&Currency::INR),
            "USD" => Some(&Currency::USD),
            "EUR" => Some(&Currency::EUR),
            "GBP" => Some(&Currency::GBP),
            "JPY" => Some(&Currency::JPY),
            "USDT" => Some(&Currency::USDT),
            _ => None,
        }
    }

    /// Format `amount` according to this currency's style.
    ///
    /// - INR uses Indian lakh/crore grouping: `₹1,23,45,678.00`
    /// - Others use standard Western grouping: `$1,234,567.89`
    pub fn format(&self, amount: f64) -> String {
        let neg = amount < 0.0;
        let abs = amount.abs();
        let integer_part = abs.floor() as u64;
        let frac_part = abs - abs.floor();

        let grouped = if self.lakh_grouping {
            format_lakh(integer_part)
        } else {
            format_standard(integer_part)
        };

        let dp = self.decimal_places as usize;
        let decimal_str = if dp > 0 {
            let frac = (frac_part * 10f64.powi(dp as i32)).round() as u64;
            format!(".{:0>width$}", frac, width = dp)
        } else {
            String::new()
        };

        let sign = if neg { "-" } else { "" };
        format!("{}{}{}{}", sign, self.symbol, grouped, decimal_str)
    }
}

/// Indian lakh/crore grouping: last 3 digits, then groups of 2 from the right.
/// e.g. 12345678 → "1,23,45,678"
fn format_lakh(n: u64) -> String {
    let s = n.to_string();
    if s.len() <= 3 {
        return s;
    }
    let (rest, last3) = s.split_at(s.len() - 3);
    let mut out = String::new();
    let chars: Vec<char> = rest.chars().collect();
    let first_len = chars.len() % 2;
    if first_len > 0 {
        out.push_str(&chars[..first_len].iter().collect::<String>());
    }
    let mut i = first_len;
    while i < chars.len() {
        if !out.is_empty() {
            out.push(',');
        }
        out.push_str(&chars[i..i + 2].iter().collect::<String>());
        i += 2;
    }
    if !out.is_empty() {
        out.push(',');
    }
    out.push_str(last3);
    out
}

/// Standard Western grouping: groups of 3 digits from the right.
/// e.g. 1234567 → "1,234,567"
fn format_standard(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inr_format() {
        assert_eq!(Currency::INR.format(123456.78), "₹1,23,456.78");
        assert_eq!(Currency::INR.format(10000000.0), "₹1,00,00,000.00");
        assert_eq!(Currency::INR.format(100.0), "₹100.00");
        assert_eq!(Currency::INR.format(-5000.0), "-₹5,000.00");
    }

    #[test]
    fn test_usd_format() {
        assert_eq!(Currency::USD.format(1234567.89), "$1,234,567.89");
        assert_eq!(Currency::USD.format(0.5), "$0.50");
    }

    #[test]
    fn test_jpy_format() {
        assert_eq!(Currency::JPY.format(1000000.0), "¥1,000,000");
    }

    #[test]
    fn test_from_code() {
        assert_eq!(Currency::from_code("inr"), Some(&Currency::INR));
        assert_eq!(Currency::from_code("USD"), Some(&Currency::USD));
        assert_eq!(Currency::from_code("UNKNOWN"), None);
    }
}
