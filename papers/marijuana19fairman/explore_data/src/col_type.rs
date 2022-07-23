use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColType {
    Int,
    Float,
    String,
}

impl ColType {
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (_, Self::String) | (Self::String, _) => Self::String,
            (_, Self::Float) | (Self::Float, _) => Self::Float,
            _ => Self::Int,
        }
    }
}

impl From<char> for ColType {
    fn from(c: char) -> Self {
        match c {
            '0'..='9' | '-' => Self::Int,
            '.' => Self::Float,
            _ => Self::String,
        }
    }
}

impl Display for ColType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColType::Int => write!(f, "int"),
            ColType::Float => write!(f, "float"),
            ColType::String => write!(f, "str"),
        }
    }
}

impl From<&str> for ColType {
    fn from(s: &str) -> Self {
        s.chars()
            .map(|c| c.into())
            .reduce(|a: ColType, b| a.combine(b))
            .unwrap_or(ColType::Int)
    }
}
