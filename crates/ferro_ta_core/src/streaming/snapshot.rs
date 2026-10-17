//! Versioned JSON snapshots of streaming state (feature `serde`).
//!
//! A snapshot is `{"format": 1, "kind": "<type tag>", "state": {...}}`.
//! Loading checks the format version and the type tag before decoding, so a
//! snapshot can never be restored into the wrong indicator or misread by a
//! newer layout. Restored state continues bit-identically.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::{BarBuilder, Live, StreamValue, StreamingError, StreamingIndicator};

/// Snapshot layout version. Bump on any change to a serialized state struct.
pub const STATE_FORMAT_VERSION: u32 = 1;

const BAR_BUILDER_KIND: &str = "bar_builder";

#[derive(Serialize)]
struct EnvelopeRef<'a, S> {
    format: u32,
    kind: &'a str,
    state: &'a S,
}

#[derive(Deserialize)]
struct Envelope {
    format: u32,
    kind: String,
    state: serde_json::Value,
}

fn snapshot_err(msg: impl std::fmt::Display) -> StreamingError {
    StreamingError(format!("invalid state snapshot: {msg}"))
}

fn save_state<S: Serialize>(kind: &str, state: &S) -> Result<String, StreamingError> {
    serde_json::to_string(&EnvelopeRef {
        format: STATE_FORMAT_VERSION,
        kind,
        state,
    })
    .map_err(snapshot_err)
}

fn load_state<S: DeserializeOwned>(kind: &str, json: &str) -> Result<S, StreamingError> {
    let envelope: Envelope = serde_json::from_str(json).map_err(snapshot_err)?;
    if envelope.format != STATE_FORMAT_VERSION {
        return Err(snapshot_err(format_args!(
            "format {} is not supported (expected {STATE_FORMAT_VERSION})",
            envelope.format
        )));
    }
    if envelope.kind != kind {
        return Err(snapshot_err(format_args!(
            "kind {:?} cannot be loaded as {kind:?}",
            envelope.kind
        )));
    }
    serde_json::from_value(envelope.state).map_err(snapshot_err)
}

impl<T> Live<T>
where
    T: StreamingIndicator + Serialize + DeserializeOwned,
{
    /// Serialize the full state (indicator, last value, bar count).
    pub fn save(&self) -> Result<String, StreamingError> {
        save_state(T::KIND, self)
    }

    /// Restore a snapshot made by [`Live::save`] for the same indicator type.
    pub fn load(json: &str) -> Result<Self, StreamingError> {
        load_state(T::KIND, json)
    }
}

impl BarBuilder {
    /// Serialize the builder, including its forming bar and tick history.
    pub fn save(&self) -> Result<String, StreamingError> {
        save_state(BAR_BUILDER_KIND, self)
    }

    /// Restore a snapshot made by [`BarBuilder::save`]; the session spec is
    /// re-validated.
    pub fn load(json: &str) -> Result<Self, StreamingError> {
        let builder: BarBuilder = load_state(BAR_BUILDER_KIND, json)?;
        BarBuilder::new(builder.spec())?;
        Ok(builder)
    }
}

/// `f64` with NaN stored as JSON `null` (serde_json cannot encode NaN).
pub(crate) mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        (!v.is_nan()).then_some(*v).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

/// A [`StreamValue`] stored as a list of numbers with NaN as `null`.
pub(crate) mod stream_value {
    use super::StreamValue;
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<V: StreamValue, S: Serializer>(v: &V, s: S) -> Result<S::Ok, S::Error> {
        v.to_parts().serialize(s)
    }

    pub fn deserialize<'de, V: StreamValue, D: Deserializer<'de>>(d: D) -> Result<V, D::Error> {
        let parts = Vec::<Option<f64>>::deserialize(d)?;
        V::from_parts(&parts).ok_or_else(|| D::Error::custom("value has the wrong shape"))
    }
}

#[cfg(test)]
mod tests {
    use crate::options::{OptionKind, PricingModel};
    use crate::streaming::*;
    use serde::{de::DeserializeOwned, Serialize};

    const WARM: usize = 150;

    /// Deterministic OHLCV path (LCG), as in the live-bar tests.
    fn bars(n: usize) -> Vec<(f64, f64, f64, f64)> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut close = 100.0;
        (0..n)
            .map(|_| {
                close += next() - 0.5;
                (close + next(), close - next(), close, 1.0 + 10.0 * next())
            })
            .collect()
    }

    fn same<O: std::fmt::Debug>(a: O, b: O) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }

    fn assert_round_trip<T>(
        make: impl Fn() -> T,
        input_of: impl Fn((f64, f64, f64, f64)) -> T::Input,
    ) where
        T: StreamingIndicator + Serialize + DeserializeOwned,
    {
        let data = bars(2 * WARM);
        let mut original = Live::new(make());
        for &bar in &data[..WARM] {
            original.update(input_of(bar), true);
        }
        let json = original.save().unwrap();
        let mut restored = Live::<T>::load(&json).unwrap();
        assert_eq!(restored.count(), original.count(), "{}", T::KIND);
        assert!(same(restored.last(), original.last()), "{}", T::KIND);
        for (i, &bar) in data[WARM..].iter().enumerate() {
            let a = original.update(input_of(bar), true);
            let b = restored.update(input_of(bar), true);
            assert!(same(a, b), "{} bar {i}: {a:?} != {b:?}", T::KIND);
        }
    }

    #[test]
    fn round_trip_is_bit_identical_for_every_indicator() {
        let close = |b: (f64, f64, f64, f64)| b.2;
        let hlc = |b: (f64, f64, f64, f64)| (b.0, b.1, b.2);
        assert_round_trip(|| StreamingSMA::new(10).unwrap(), close);
        assert_round_trip(|| StreamingEMA::new(10).unwrap(), close);
        assert_round_trip(|| StreamingRSI::new(14).unwrap(), close);
        assert_round_trip(|| StreamingBBands::new(20, 2.0, 2.0).unwrap(), close);
        assert_round_trip(|| StreamingMACD::new(12, 26, 9).unwrap(), close);
        assert_round_trip(|| StreamingATR::new(14).unwrap(), hlc);
        assert_round_trip(|| StreamingStoch::new(5, 3, 3).unwrap(), hlc);
        assert_round_trip(|| StreamingSupertrend::new(7, 3.0).unwrap(), hlc);
        assert_round_trip(StreamingVWAP::new, |b| b);
        assert_round_trip(|| StreamingCross::new(0.1).unwrap(), |b| (b.2, b.0 - 0.5));
        assert_round_trip(|| StreamingStreak::new(3).unwrap(), |b| b.2 - b.1 - 0.5);
        assert_round_trip(|| StreamingCooldown::new(2).unwrap(), |b| b.2 - 100.0);
        assert_round_trip(
            || {
                StreamingOptionGreeks::new(
                    PricingModel::BlackScholes,
                    OptionKind::Call,
                    100.0,
                    0.05,
                    0.0,
                )
                .unwrap()
            },
            |b| (5.0 + (b.0 - b.1), b.2, 0.25),
        );
    }

    #[test]
    fn warmup_nan_state_round_trips() {
        let mut ema = Live::new(StreamingEMA::new(5).unwrap());
        ema.update(1.0, true);
        let restored = Live::<StreamingEMA>::load(&ema.save().unwrap()).unwrap();
        assert!(restored.last().is_nan() && !restored.is_ready());

        let greeks = Live::new(
            StreamingOptionGreeks::new(
                PricingModel::BlackScholes,
                OptionKind::Put,
                100.0,
                0.0,
                0.0,
            )
            .unwrap(),
        );
        let restored = Live::<StreamingOptionGreeks>::load(&greeks.save().unwrap()).unwrap();
        assert!(restored.inner().last_iv().is_nan());
    }

    #[test]
    fn saved_state_is_versioned_envelope() {
        let sma = Live::new(StreamingSMA::new(3).unwrap());
        let value: serde_json::Value = serde_json::from_str(&sma.save().unwrap()).unwrap();
        assert_eq!(value["format"], STATE_FORMAT_VERSION);
        assert_eq!(value["kind"], "sma");
        assert!(value["state"].is_object());
    }

    #[test]
    fn load_rejects_wrong_kind_and_version() {
        let json = Live::new(StreamingSMA::new(3).unwrap()).save().unwrap();
        assert!(Live::<StreamingEMA>::load(&json).is_err(), "kind mismatch");
        let future = json.replace("\"format\":1", "\"format\":2");
        assert_ne!(future, json);
        assert!(
            Live::<StreamingSMA>::load(&future).is_err(),
            "future format"
        );
        assert!(Live::<StreamingSMA>::load("not json").is_err());
        assert!(
            Live::<StreamingSMA>::load("{\"format\":1,\"kind\":\"sma\",\"state\":{}}").is_err()
        );
    }

    #[test]
    fn bar_builder_round_trip_keeps_forming_bar() {
        let ist = |m: i64| 20_377 * 86_400_000 + (9 * 60 + m) * 60_000 - 19_800_000;
        let mut b = BarBuilder::new(BarSpec::nse(300)).unwrap();
        b.on_tick(ist(16), 100.0, 1.0);
        b.on_tick(ist(17), 103.0, 2.0);
        let mut restored = BarBuilder::load(&b.save().unwrap()).unwrap();
        assert_eq!(restored.forming(), b.forming());
        assert_eq!(
            restored.on_tick(ist(17), 103.0, 2.0).0,
            TickStatus::Duplicate
        );
        assert_eq!(
            restored.on_tick(ist(20), 99.0, 1.0),
            b.on_tick(ist(20), 99.0, 1.0)
        );
        let json = b.save().unwrap();
        assert!(
            Live::<StreamingSMA>::load(&json).is_err(),
            "bar_builder is not sma"
        );
    }
}
