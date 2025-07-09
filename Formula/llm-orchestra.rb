class LlmOrchestra < Formula
  include Language::Python::Virtualenv

  desc "Multi-agent LLM communication system with ensemble orchestration"
  homepage "https://github.com/mrilikecoding/llm-orc"
  url "https://github.com/mrilikecoding/llm-orc/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "10121e0e961cf8aeab81da8a577a4dbd1b48a25fe205b701743cc518db9bda0b"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources using: "python@3.12"
  end

  test do
    system "#{bin}/llm-orc", "--help"
    assert_match "llm orchestra", shell_output("#{bin}/llm-orc --help").downcase
  end
end