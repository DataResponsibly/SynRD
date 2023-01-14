use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColType {
    Int,
    Float,
    String,
}

impl ColType {
    pub fn combine(&self, other: &Self) -> Self {
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
        s.chars().fold(ColType::Int, |a, b| a.combine(&b.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion() {
        assert_eq!(ColType::from("123"), ColType::Int);
        assert_eq!(ColType::from("0.9"), ColType::Float);
        assert_eq!(ColType::from("03/19/2019"), ColType::String);
    }
}
