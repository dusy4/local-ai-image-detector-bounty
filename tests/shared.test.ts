import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { fuseScores } from "../src/shared.ts";
import { inspectMetadata } from "../src/metadata.ts";

describe("score fusion", () => {
  it("does not amplify weak signals", () => assert.equal(fuseScores(0.4, [{ source: "x", score: 0.5, detail: "weak" }]), 0.4));
  it("raises scores for strong provenance", () => assert.ok(fuseScores(0.4, [{ source: "x", score: 0.99, detail: "strong" }]) > 0.99));
  it("ignores shadow experts", () => assert.equal(fuseScores(0.4, [], [{ source: "x", score: 0.99, active: false, detail: "shadow" }]), 0.4));
  it("accepts calibrated active experts", () => assert.ok(fuseScores(0.4, [], [{ source: "x", score: 0.9, active: true, detail: "validated" }]) > 0.9));
  it("implements the validated threshold OR without combining sub-threshold experts", () => {
    assert.equal(fuseScores(0.64, [], [{ source: "x", score: 0.64, active: true, detail: "weak" }]), 0.64);
    assert.ok(fuseScores(0.2, [], [{ source: "x", score: 0.65, active: true, detail: "validated" }]) >= 0.65);
  });
  it("applies the locked three-model calibration", () => {
    const experts = [
      { source: "Community Forensics", score: 0.8, active: true, detail: "validated" },
      { source: "Modern ConvNeXt", score: 0.8, active: true, detail: "validated" },
    ];
    assert.ok(fuseScores(0.8, [], experts) > 0.65);
    assert.ok(fuseScores(0.01, [], experts.map(expert => ({ ...expert, score: 0.01 }))) < 0.65);
  });
});
describe("metadata", () => {
  it("requires cryptographic validation before trusting C2PA AI assertions", () => assert.ok(inspectMetadata(new TextEncoder().encode("digitalSourceType=trainedAlgorithmicMedia"))[0].score < 0.65));
  it("does not call generic C2PA AI", () => assert.ok(inspectMetadata(new TextEncoder().encode("c2pa camera credential"))[0].score < 0.65));
});
