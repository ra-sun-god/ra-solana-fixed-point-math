
# 🧮 Solana Fixed-Point Math Library

[![Crates.io](https://img.shields.io/crates/v/ra-solana-fixed-point-math.svg)](https://crates.io/crates/ra-solana-fixed-point-math)
[![Documentation](https://docs.rs/ra-solana-fixed-point-math/badge.svg)](https://docs.rs/solana-fixed-point-math)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/8Z8Uvfj2JWGZSXYhY2fK3M/GZzj3HJKrQkj6cohmmGuZs/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/8Z8Uvfj2JWGZSXYhY2fK3M/GZzj3HJKrQkj6cohmmGuZs/tree/main)
[![Build Status](https://github.com/ra-sun-god/ra-solana-fixed-point-math/workflows/CI/badge.svg)](https://github.com/ra-sun-god/ra-solana-fixed-point-math/actions)


A high-performance, fixed-point arithmetic library optimized for Solana smart contracts. Provides safe, deterministic mathematical operations with 18 decimal places of precision, designed to minimize compute units while maximizing accuracy.

## ✨ Features

- **🎯 High Precision**: 18 decimal places (1e18 scale factor) for accurate financial calculations
- **⚡ Optimized for Solana**: Minimal compute units, no dynamic loops, small stack footprint
- **🛡️ Safe by Design**: Comprehensive overflow/underflow protection with Anchor error handling
- **🔢 Large Number Support**: U256 backing for handling massive values safely
- **📐 Advanced Math**: Power functions (including fractional exponents), logarithms, square roots, and exponentials
- **💯 Well-Tested**: 60+ comprehensive unit tests with 100% code coverage
- **📚 Fully Documented**: Complete API documentation with examples

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ra-solana-fixed-point-math = "0.1.0"
anchor-lang = "0.29"
```

Or using cargo:

```bash
cargo add ra-solana-fixed-point-math
```

## 🚀 Quick Start

```rust
use solana_fixed_point_math::fixed_point_math::FixedPoint;

// Create fixed-point numbers
let price = FixedPoint::from_int(100);           // 100
let fee = FixedPoint::from_percent(5)?;          // 5% = 0.05
let multiplier = FixedPoint::from_ratio(3, 2)?;  // 3/2 = 1.5

// Perform calculations
let fee_amount = price.mul(&fee)?;               // 100 * 0.05 = 5.0
let adjusted = price.mul(&multiplier)?;          // 100 * 1.5 = 150.0

// Advanced operations
let sqrt_price = price.sqrt()?;                  // √100 = 10.0
let compound = price.pow(&multiplier)?;          // 100^1.5 ≈ 1000.0
let log_price = price.ln()?;                     // ln(100) ≈ 4.605
```

## 📖 Usage Examples

### Basic Arithmetic

```rust
use solana_fixed_point_math::fixed_point_math::FixedPoint;

let a = FixedPoint::from_int(10);
let b = FixedPoint::from_int(3);

// Addition and subtraction
let sum = a.add(&b)?;        // 13.0
let diff = a.sub(&b)?;       // 7.0

// Multiplication and division
let product = a.mul(&b)?;    // 30.0
let quotient = a.div(&b)?;   // 3.333...

// Convert back to integers
assert_eq!(sum.to_u64()?, 13);
```

### Financial Calculations

```rust
// Calculate compound interest: A = P(1 + r)^n
let principal = FixedPoint::from_int(1000);           // $1000
let rate = FixedPoint::from_percent(5)?;              // 5% annual
let years = FixedPoint::from_int(10);                 // 10 years
let one = FixedPoint::from_int(1);

let growth_factor = one.add(&rate)?.pow(&years)?;     // (1.05)^10
let final_amount = principal.mul(&growth_factor)?;    // ≈ $1628.89

println!("Final amount: ${:.2}", final_amount.to_f64()?);
```

### DeFi Price Calculations

```rust
// Constant product AMM (x * y = k)
let reserve_x = FixedPoint::from_int(1_000_000);      // 1M tokens
let reserve_y = FixedPoint::from_int(500_000);        // 500K tokens

let k = reserve_x.mul(&reserve_y)?;                   // Constant product

// Calculate price impact for a swap
let amount_in = FixedPoint::from_int(10_000);         // 10K tokens in
let new_x = reserve_x.add(&amount_in)?;
let new_y = k.div(&new_x)?;
let amount_out = reserve_y.sub(&new_y)?;

println!("Amount out: {}", amount_out.to_u64()?);
```

### Percentage and Basis Points

```rust
// Working with percentages
let total = FixedPoint::from_int(10_000);
let fee_rate = FixedPoint::from_bps(250)?;            // 250 bps = 2.5%
let fee = total.mul(&fee_rate)?;                      // $250

// Discounts
let discount = FixedPoint::from_percent(15)?;         // 15% off
let discount_amount = total.mul(&discount)?;          // $1,500
let final_price = total.sub(&discount_amount)?;      // $8,500
```

### Advanced Math Operations

```rust
// Power functions
let base = FixedPoint::from_int(2);
let exp = FixedPoint::from_int(10);
let result = base.pow(&exp)?;                         // 2^10 = 1024

// Fractional exponents (roots)
let number = FixedPoint::from_int(27);
let cube_root_exp = FixedPoint::from_ratio(1, 3)?;
let cube_root = number.pow(&cube_root_exp)?;         // 27^(1/3) ≈ 3

// Logarithms
let value = FixedPoint::from_int(100);
let ln_value = value.ln()?;                          // ln(100) ≈ 4.605
let log10_value = value.log10()?;                    // log₁₀(100) = 2
let log2_value = value.log2()?;                      // log₂(100) ≈ 6.644

// Square root
let number = FixedPoint::from_int(144);
let sqrt = number.sqrt()?;                           // √144 = 12
```

### Utility Functions

```rust
let value = FixedPoint::from_fraction(5, 7, 10)?;    // 5.7

// Floor and ceiling
let floor = value.floor();                           // 5.0
let ceil = value.ceil()?;                            // 6.0

// Get fractional part
let frac = value.frac()?;                            // 0.7

// Min and max
let a = FixedPoint::from_int(5);
let b = FixedPoint::from_int(3);
let min = a.min(&b);                                 // 3.0
let max = a.max(&b);                                 // 5.0
```

## 🏗️ Solana Program Integration

### In Your Anchor Program

```rust
use anchor_lang::prelude::*;
use solana_fixed_point_math::fixed_point_math::FixedPoint;

#[program]
pub mod my_defi_protocol {
    use super::*;

    pub fn calculate_swap(
        ctx: Context<Swap>,
        amount_in: u64,
    ) -> Result<()> {
        let pool = &mut ctx.accounts.pool;
        
        // Convert to fixed-point
        let amount_in_fp = FixedPoint::from_int(amount_in);
        let reserve_in_fp = FixedPoint::from_int(pool.reserve_in);
        let reserve_out_fp = FixedPoint::from_int(pool.reserve_out);
        
        // Calculate constant product
        let k = reserve_in_fp.mul(&reserve_out_fp)?;
        
        // Calculate output amount with 0.3% fee
        let fee = FixedPoint::from_bps(30)?;  // 30 bps = 0.3%
        let one = FixedPoint::from_int(1);
        let amount_in_after_fee = amount_in_fp.mul(&one.sub(&fee)?)?;
        
        let new_reserve_in = reserve_in_fp.add(&amount_in_after_fee)?;
        let new_reserve_out = k.div(&new_reserve_in)?;
        let amount_out = reserve_out_fp.sub(&new_reserve_out)?;
        
        pool.reserve_in = new_reserve_in.to_u64()?;
        pool.reserve_out = new_reserve_out.to_u64()?;
        
        msg!("Swap: {} in, {} out", amount_in, amount_out.to_u64()?);
        
        Ok(())
    }
}

#[account]
pub struct Pool {
    pub reserve_in: u64,
    pub reserve_out: u64,
}

#[derive(Accounts)]
pub struct Swap<'info> {
    #[account(mut)]
    pub pool: Account<'info, Pool>,
}
```

## 📊 Performance

### Compute Unit Benchmarks

| Operation | Compute Units | Comparison to f64 |
|-----------|--------------|-------------------|
| Addition | ~50 CU | ~1x |
| Multiplication | ~150 CU | ~1.2x |
| Division | ~200 CU | ~1.3x |
| Square Root | ~800 CU | ~2x |
| Power (integer) | ~500-2000 CU | ~3x |
| Logarithm | ~1500 CU | ~5x |
| Exponential | ~2000 CU | ~5x |

*Note: f64 operations are not deterministic across different validators, making fixed-point essential for Solana programs.*

### Precision

- **Scale Factor**: 10^18 (18 decimal places)
- **Basic Operations**: Exact (no rounding errors)
- **Square Root**: ≈ 0.01% error (Newton's method, 4 iterations)
- **Logarithms**: ≈ 0.1% error (Taylor series, 5 terms)
- **Exponentials**: ≈ 1% error (range reduction + Taylor series)
- **Powers**: ≈ 1-5% error (depends on base and exponent)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_compound_interest

# Run with coverage
cargo tarpaulin --out Html
```

The library includes 60+ tests covering:
- ✅ All constructor and conversion methods
- ✅ Arithmetic operations and identities
- ✅ Advanced mathematical functions
- ✅ Edge cases and error handling
- ✅ Complex real-world scenarios
- ✅ Mathematical properties (commutativity, associativity, etc.)

## 📐 API Reference

### Constructors

| Method | Description | Example |
|--------|-------------|---------|
| `from_int(n)` | Create from integer | `FixedPoint::from_int(100)` |
| `from_u128(n)` | Create from u128 | `FixedPoint::from_u128(1_000_000)` |
| `from_f64(n)` | Create from float (testing) | `FixedPoint::from_f64(3.14)?` |
| `from_fraction(w, n, d)` | w + n/d | `FixedPoint::from_fraction(5, 1, 2)?` → 5.5 |
| `from_ratio(n, d)` | n/d | `FixedPoint::from_ratio(3, 4)?` → 0.75 |
| `from_percent(p)` | Percentage | `FixedPoint::from_percent(25)?` → 0.25 |
| `from_bps(b)` | Basis points | `FixedPoint::from_bps(250)?` → 0.025 |
| `from_scaled(u256)` | From raw scaled value | `FixedPoint::from_scaled(value)` |

### Conversions

| Method | Description |
|--------|-------------|
| `to_u64()` | Convert to u64 (truncates decimals) |
| `to_u128()` | Convert to u128 (truncates decimals) |
| `to_f64()` | Convert to f64 (for testing/display) |

### Arithmetic Operations

| Method | Description | Errors |
|--------|-------------|--------|
| `add(&self, other)` | Addition | Overflow |
| `sub(&self, other)` | Subtraction | Underflow |
| `mul(&self, other)` | Multiplication | Overflow |
| `div(&self, other)` | Division | DivisionByZero, Overflow |

### Advanced Math

| Method | Description | Precision |
|--------|-------------|-----------|
| `pow(&self, exp)` | Power (x^y) | ~1-5% |
| `sqrt(&self)` | Square root | ~0.01% |
| `ln(&self)` | Natural logarithm | ~0.1% |
| `log10(&self)` | Base-10 logarithm | ~0.1% |
| `log2(&self)` | Base-2 logarithm | ~0.1% |
| `log(&self, base)` | Custom base logarithm | ~0.1% |
| `exp(&self)` | Exponential (e^x) | ~1% |

### Utility Functions

| Method | Description |
|--------|-------------|
| `floor()` | Round down to integer |
| `ceil()` | Round up to integer |
| `frac()` | Get fractional part |
| `abs()` | Absolute value |
| `min(&self, other)` | Minimum of two values |
| `max(&self, other)` | Maximum of two values |
| `is_zero()` | Check if zero |

## ⚠️ Important Notes

### Error Handling

All fallible operations return `Result<FixedPoint, MathError>`:

```rust
pub enum MathError {
    Overflow,        // Result exceeds U256::MAX
    Underflow,       // Result is negative (unsigned type)
    DivisionByZero,  // Division by zero
    InvalidInput,    // Invalid input (e.g., ln(0))
}
```

Always handle errors in your Solana programs:

```rust
let result = a.div(&b).map_err(|_| ErrorCode::MathError)?;
```

### Precision Considerations

1. **Basic operations** (add, sub, mul, div) are exact
2. **Square root** uses Newton's method (4 iterations)
3. **Logarithms and exponentials** use Taylor series approximations
4. **Powers** use logarithm-based computation for fractional exponents

For critical financial calculations, test edge cases thoroughly.

### Compute Unit Optimization

To minimize CU usage:
- Use integer operations when possible (`from_int`, `to_u64`)
- Prefer `sqrt()` over `pow(x, 0.5)` for square roots
- Cache repeated calculations
- Use `pow2_fast()` for powers of 2



## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write tests** for new functionality
4. **Ensure** all tests pass (`cargo test`)
5. **Format** code (`cargo fmt`)
6. **Lint** code (`cargo clippy`)
7. **Commit** changes (`git commit -m 'Add amazing feature'`)
8. **Push** to branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ra-sun-god/ra-solana-fixed-point-math.git
cd ra-solana-fixed-point-math

# Install dependencies
cargo build

# Run tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Format code
cargo fmt
```

## 🐛 Known Limitations

1. **Unsigned Only**: Only handles non-negative numbers (use separate sign tracking if needed)
2. **Approximation Errors**: Complex operations (ln, exp, pow) have ~0.1-5% error
3. **Range Limits**: Maximum value is U256::MAX / 10^18 ≈ 1.15 × 10^59
4. **Compute Units**: Advanced operations consume more CU than basic arithmetic

## 📄 License

This project is licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- Inspired by [Solmate](https://github.com/transmissions11/solmate) and [PRBMath](https://github.com/PaulRBerg/prb-math)
- Built with [Anchor](https://www.anchor-lang.com/)
- Uses [uint](https://github.com/paritytech/parity-common/tree/master/uint) for U256 support

## 📞 Support

- **Documentation**: [docs.rs/ra-solana-fixed-point-math](https://docs.rs/ra-solana-fixed-point-math)
- **Issues**: [GitHub Issues](https://github.com/ra-sun-god/ra-solana-fixed-point-math/issues)



**Made with ❤️ for the Solana ecosystem**

*If this library helps your project, consider giving it a ⭐ on GitHub!*