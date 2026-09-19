import { describe, expect, test } from "@jest/globals"
import { poseidon2HashAsync } from "@zkpassport/poseidon2"
import type { IntegrityToDisclosureSalts, PackagedCertificatesFile, Query } from "@zkpassport/utils"
import {
  Binary,
  DisclosedData,
  calculatePackagedCertificatesRoot,
  getAgeCircuitInputs,
  getBindCircuitInputs,
  getBirthdateCircuitInputs,
  getCertificateRegistryRootFromOuterProof,
  getCircuitMerkleProof,
  getCommitmentFromDSCProof,
  getCommitmentInFromDisclosureProof,
  getCommitmentInFromIDDataProof,
  getCommitmentInFromIntegrityProof,
  getCommitmentOutFromIDDataProof,
  getCommitmentOutFromIntegrityProof,
  getCurrentDateFromOuterProof,
  getDiscloseCircuitInputs,
  getDiscloseEVMParameterCommitment,
  getDiscloseParameterCommitment,
  getDisclosedBytesFromMrzAndMask,
  getExpiryDateCircuitInputs,
  getIssuingCountryExclusionCircuitInputs,
  getIssuingCountryInclusionCircuitInputs,
  getMerkleRootFromDSCProof,
  getNationalityExclusionCircuitInputs,
  getNationalityInclusionCircuitInputs,
  getNowTimestamp,
  getNullifierFromDisclosureProof,
  getNullifierFromOuterProof,
  getNullifierTypeFromDisclosureProof,
  getNullifierTypeFromOuterProof,
  NullifierType,
  getScopeFromOuterProof,
  getSubscopeFromOuterProof,
  getSanctionsExclusionCheckCircuitInputs,
  getOuterCircuitInputs,
  getParamCommitmentsFromOuterProof,
  getParameterCommitmentFromDisclosureProof,
  getServiceScopeHash,
  getServiceSubscopeHash,
  ProofType,
  getFacematchCircuitInputs,
  packLeBytesAndHashPoseidon2,
  getFacematchEvmParameterCommitment,
  getOprfPkHashFromOuterProof,
} from "@zkpassport/utils"
import * as path from "path"
import * as fs from "fs"
import { Circuit } from "../circuits"
import { generateSigningCertificates, loadKeypairFromFile, signSod } from "../passport-generator"
import { generateSod, wrapSodInContentInfo } from "../sod-generator"
import { TestHelper, convertPemToPackagedCertificateV1 } from "../test-helper"
import { createUTCDate, serializeAsn } from "../utils"
const { calculateCircuitRoot } = require("@zkpassport/utils/registry")
import FIXTURES_FACEMATCH from "./fixtures/facematch"
import { AlgorithmIdentifier } from "@peculiar/asn1-x509"
import { id_sha256WithRSAEncryption } from "@peculiar/asn1-rsa"

const nowTimestamp = getNowTimestamp()
const INTEGRITY_TO_DISCLOSURE_SALTS: IntegrityToDisclosureSalts = {
  dg1Salt: 3n,
  expiryDateSalt: 3n,
  dg2HashSalt: 3n,
  privateNullifierSalt: 3n,
}

describe("outer proof", () => {
  const helper = new TestHelper()
  const packagedCerts: PackagedCertificatesFile = { version: 1, timestamp: 0, root: "", certificates: [], masterlists: [], revocations: [] }
  const FIXTURES_PATH = path.join(__dirname, "fixtures")
  const DSC_KEYPAIR_PATH = path.join(FIXTURES_PATH, "dsc-keypair-rsa.json")
  const MAX_TBS_LENGTH = 700
  let subproofs: Map<
    number,
    {
      proof: string[]
      publicInputs: string[]
      vkey: string[]
      vkeyHash: string
      paramCommitment?: bigint
    }
  > = new Map()
  let certificateRegistryRoot: bigint

  beforeEach(async () => {
    // Johnny Silverhand's MRZ
    const mrz =
    "P<AUSSILVERHAND<<JOHNNY<<<<<<<<<<<<<<<<<<<<<PA1234567_AUS881112_M300101_<CYBERCITY<<<<\0\0"
    const dg1 = Binary.fromHex("615B5F1F58").concat(Binary.from(mrz))
    // Load DSC keypair
    const dscKeypair = await loadKeypairFromFile(DSC_KEYPAIR_PATH)

    // Generate CSC and DSC signing certificates
    const { cscPem, dsc, dscKeys } = await generateSigningCertificates({
      cscSigningHashAlgorithm: "SHA-256",
      cscKeyType: "RSA",
      cscKeySize: 2048,
      dscSigningHashAlgorithm: "SHA-256",
      dscKeyType: "RSA",
      dscKeySize: 2048,
      dscKeypair,
    })
    // Generate SOD and sign it with DSC keypair
    const { sod } = await generateSod(dg1, [dsc], "SHA-256", new AlgorithmIdentifier({
      algorithm: id_sha256WithRSAEncryption,
    }))
    const { sod: signedSod } = await signSod(sod, dscKeys, "SHA-256")
    // Add newly generated CSC to masterlist
    packagedCerts.certificates.push(await convertPemToPackagedCertificateV1(cscPem))
    packagedCerts.timestamp = Math.floor(Date.UTC(2026, 0, 1) / 1000)
    packagedCerts.root = await calculatePackagedCertificatesRoot(packagedCerts)
    // Load passport data into helper
    const contentInfoWrappedSod = serializeAsn(wrapSodInContentInfo(signedSod))
    await helper.loadPassport(dg1, Binary.from(contentInfoWrappedSod))
    helper.setCertificates(packagedCerts)

    subproofs = new Map()
    const cscToDscCircuit = Circuit.from(`sig_check_dsc_tbs_${MAX_TBS_LENGTH}_rsa_pkcs_2048_sha256`)
    const cscToDscInputs = await helper.generateCircuitInputs("dsc")
    const cscToDscProof = await cscToDscCircuit.prove(cscToDscInputs, {
      recursive: true,
      useCli: true,
      circuitName: `sig_check_dsc_tbs_${MAX_TBS_LENGTH}_rsa_pkcs_2048_sha256`,
    })
    expect(cscToDscProof).toBeDefined()
    expect(cscToDscProof.publicInputs.length).toEqual(2)
    certificateRegistryRoot = getMerkleRootFromDSCProof(cscToDscProof)
    expect(certificateRegistryRoot).toBeDefined()
    const cscToDscCommitment = getCommitmentFromDSCProof(cscToDscProof)
    const cscToDscVkey = (await cscToDscCircuit.getVerificationKey({ evm: false })).vkeyFields
    const cscToDscVkeyHash = `0x${(
      await poseidon2HashAsync(cscToDscVkey.map((x) => BigInt(x)))
    ).toString(16)}`
    subproofs.set(0, {
      proof: cscToDscProof.proof,
      publicInputs: cscToDscProof.publicInputs,
      vkey: cscToDscVkey,
    vkeyHash: cscToDscVkeyHash,
    })
    await cscToDscCircuit.destroy()

    const idDataToIntegrityCircuit = Circuit.from(
    `sig_check_id_data_tbs_${MAX_TBS_LENGTH}_rsa_pkcs_2048_sha256`,
    )
    const idDataToIntegrityInputs = await helper.generateCircuitInputs("id")
    const idDataToIntegrityProof = await idDataToIntegrityCircuit.prove(idDataToIntegrityInputs, {
      recursive: true,
      useCli: true,
      circuitName: `sig_check_id_data_tbs_${MAX_TBS_LENGTH}_rsa_pkcs_2048_sha256`,
    })
    expect(idDataToIntegrityProof).toBeDefined()
    const idDataCommitmentIn = getCommitmentInFromIDDataProof(idDataToIntegrityProof)
    const dscToIdDataCommitment = getCommitmentOutFromIDDataProof(idDataToIntegrityProof)
    expect(idDataCommitmentIn).toEqual(cscToDscCommitment)
    const idDataToIntegrityVkey = (
      await idDataToIntegrityCircuit.getVerificationKey({ evm: false })
    ).vkeyFields
    const idDataToIntegrityVkeyHash = `0x${(
      await poseidon2HashAsync(idDataToIntegrityVkey.map((x) => BigInt(x)))
    ).toString(16)}`
      subproofs.set(1, {
      proof: idDataToIntegrityProof.proof,
      publicInputs: idDataToIntegrityProof.publicInputs,
      vkey: idDataToIntegrityVkey,
      vkeyHash: idDataToIntegrityVkeyHash,
    })
    await idDataToIntegrityCircuit.destroy()

    const integrityCircuit = Circuit.from("data_check_integrity_sa_sha256_dg_sha256")
    const integrityInputs = await helper.generateCircuitInputs("integrity", nowTimestamp)
    const integrityProof = await integrityCircuit.prove(integrityInputs, {
      recursive: true,
      useCli: true,
      circuitName: `data_check_integrity_sa_sha256_dg_sha256`,
    })
    expect(integrityProof).toBeDefined()
    const integrityCheckCommitmentIn = getCommitmentInFromIntegrityProof(integrityProof)
    const integrityCheckToDisclosureCommitment = getCommitmentOutFromIntegrityProof(integrityProof)
    expect(integrityCheckCommitmentIn).toEqual(dscToIdDataCommitment)
    const integrityVkey = (await integrityCircuit.getVerificationKey({ evm: false })).vkeyFields
    const integrityVkeyHash = `0x${(
      await poseidon2HashAsync(integrityVkey.map((x) => BigInt(x)))
    ).toString(16)}`
      subproofs.set(2, {
      proof: integrityProof.proof,
      publicInputs: integrityProof.publicInputs,
      vkey: integrityVkey,
      vkeyHash: integrityVkeyHash,
    })
    await integrityCircuit.destroy()

    const discloseCircuit = Circuit.from("disclose_bytes")
    const query: Query = {
      issuing_country: { disclose: true },
      nationality: { disclose: true },
      document_type: { disclose: true },
      document_number: { disclose: true },
      fullname: { disclose: true },
      birthdate: { disclose: true },
      expiry_date: { disclose: true },
      gender: { disclose: true },
    }
    let inputs = await getDiscloseCircuitInputs(helper.passport as any, query, INTEGRITY_TO_DISCLOSURE_SALTS, 0n, 0n, 0n, nowTimestamp)
    if (!inputs) throw new Error("Unable to generate disclose circuit inputs")
    const proof = await discloseCircuit.prove(inputs, {
      recursive: true,
      useCli: true,
      circuitName: `disclose_bytes`,
    })
    expect(proof).toBeDefined()
    const paramCommitment = getParameterCommitmentFromDisclosureProof(proof)
    const disclosedBytes = getDisclosedBytesFromMrzAndMask(
      helper.passport.mrz,
      inputs.disclose_mask,
    )
    const calculatedParamCommitment = await getDiscloseParameterCommitment(
      inputs.disclose_mask,
      disclosedBytes,
    )
    expect(paramCommitment).toEqual(calculatedParamCommitment)
    // Verify the disclosed data
    const disclosedData = DisclosedData.fromDisclosedBytes(disclosedBytes, "passport")
    const nullifier = getNullifierFromDisclosureProof(proof)
    expect(disclosedData.issuingCountry).toBe("AUS")
    expect(disclosedData.nationality).toBe("AUS")
    expect(disclosedData.documentType).toBe("passport")
    expect(disclosedData.documentNumber).toBe("PA1234567")
    expect(disclosedData.name).toBe("JOHNNY SILVERHAND")
    expect(disclosedData.firstName).toBe("JOHNNY")
    expect(disclosedData.lastName).toBe("SILVERHAND")
    expect(disclosedData.dateOfBirth).toEqual(createUTCDate(1988, 10, 12))
    expect(disclosedData.dateOfExpiry).toEqual(createUTCDate(2030, 0, 1))
    expect(disclosedData.gender).toBe("M")
    expect(nullifier).toEqual(
    2650684516642119190462868389024749567829027632273482260700375874186000116367n,
    )
    const discloseCommitmentIn = getCommitmentInFromDisclosureProof(proof)
    expect(discloseCommitmentIn).toEqual(integrityCheckToDisclosureCommitment)
    const discloseVkey = (await discloseCircuit.getVerificationKey({ evm: false })).vkeyFields
    const discloseVkeyHash = `0x${(
    await poseidon2HashAsync(discloseVkey.map((x) => BigInt(x)))
    ).toString(16)}`
    subproofs.set(3, {
      proof: proof.proof,
      publicInputs: proof.publicInputs,
      vkey: discloseVkey,
      vkeyHash: discloseVkeyHash,
      paramCommitment: paramCommitment,
    })
    await discloseCircuit.destroy()
  }, 60000 * 3)
  test(
    "6 subproofs",
    async () => {
      // 2nd disclosure proof
      const nationalityCircuit = Circuit.from("compare_expiry")
      const nationalityQuery: Query = {
        expiry_date: { gte: new Date(2025, 0, 1) },
      }
      const nationalityInputs = await getExpiryDateCircuitInputs(
        helper.passport as any,
        nationalityQuery,
        INTEGRITY_TO_DISCLOSURE_SALTS,
        0n,
        0n,
        0n,
        nowTimestamp,
      )
      if (!nationalityInputs) throw new Error("Unable to generate inclusion check circuit inputs")
      const nationalityProof = await nationalityCircuit.prove(nationalityInputs, {
        recursive: true,
        useCli: true,
        circuitName: `compare_expiry`,
      })
      expect(nationalityProof).toBeDefined()
      const nationalityParamCommitment = getParameterCommitmentFromDisclosureProof(nationalityProof)
      const nationalityVkey = (await nationalityCircuit.getVerificationKey({ evm: false }))
        .vkeyFields
      const nationalityVkeyHash = `0x${(
        await poseidon2HashAsync(nationalityVkey.map((x) => BigInt(x)))
      ).toString(16)}`
      await nationalityCircuit.destroy()

      // 3rd disclosure proof
      const query: Query = {
        age: { gte: 18 },
      }
      const ageCircuit = Circuit.from("compare_age")
      const ageInputs = await getAgeCircuitInputs(
        helper.passport as any,
        query,
        INTEGRITY_TO_DISCLOSURE_SALTS,
        0n,
        0n,
        0n,
        nowTimestamp,
      )
      if (!ageInputs) throw new Error("Unable to generate compare-age greater than circuit inputs")
      const ageProof = await ageCircuit.prove(ageInputs, {
        recursive: true,
        useCli: true,
        circuitName: `compare_age`,
      })
      expect(ageProof).toBeDefined()
      const ageParamCommitment = getParameterCommitmentFromDisclosureProof(ageProof)
      const ageVkey = (await ageCircuit.getVerificationKey({ evm: false })).vkeyFields
      const ageVkeyHash = `0x${(await poseidon2HashAsync(ageVkey.map((x) => BigInt(x)))).toString(
        16,
      )}`
      await ageCircuit.destroy()

      // Outer proof: a manifest of these six circuits (the fixture manifest holds other builds)
      const hashes = [0, 1, 2, 3].map((i) => subproofs.get(i)?.vkeyHash as string).concat([nationalityVkeyHash, ageVkeyHash])
      const circuitManifest: any = { version: "bench", root: await calculateCircuitRoot({ hashes }), circuits: {} }
      hashes.forEach((hash, i) => { circuitManifest.circuits[`c${i}`] = { hash, cid: "", size: 0 } })
      const outerProofCircuit = Circuit.from("outer_count_6")
      const { path: cscToDscTreeHashPath, index: cscToDscTreeIndex } = await getCircuitMerkleProof(
        subproofs.get(0)?.vkeyHash as string,
        circuitManifest,
      )
      const { path: idDataToIntegrityTreeHashPath, index: idDataToIntegrityTreeIndex } =
        await getCircuitMerkleProof(subproofs.get(1)?.vkeyHash as string, circuitManifest)
      const { path: integrityCheckTreeHashPath, index: integrityCheckTreeIndex } =
        await getCircuitMerkleProof(subproofs.get(2)?.vkeyHash as string, circuitManifest)
      const { path: discloseTreeHashPath, index: discloseTreeIndex } = await getCircuitMerkleProof(
        subproofs.get(3)?.vkeyHash as string,
        circuitManifest,
      )
      const { path: nationalityTreeHashPath, index: nationalityTreeIndex } =
        await getCircuitMerkleProof(nationalityVkeyHash as string, circuitManifest)
      const { path: ageTreeHashPath, index: ageTreeIndex } = await getCircuitMerkleProof(
        ageVkeyHash as string,
        circuitManifest,
      )
      const inputs = await getOuterCircuitInputs(
        {
          proof: subproofs.get(0)?.proof as string[],
          publicInputs: subproofs.get(0)?.publicInputs as string[],
          vkey: subproofs.get(0)?.vkey as string[],
          keyHash: subproofs.get(0)?.vkeyHash as string,
          treeHashPath: cscToDscTreeHashPath,
          treeIndex: cscToDscTreeIndex.toString(),
        },
        {
          proof: subproofs.get(1)?.proof as string[],
          publicInputs: subproofs.get(1)?.publicInputs as string[],
          vkey: subproofs.get(1)?.vkey as string[],
          keyHash: subproofs.get(1)?.vkeyHash as string,
          treeHashPath: idDataToIntegrityTreeHashPath,
          treeIndex: idDataToIntegrityTreeIndex.toString(),
        },
        {
          proof: subproofs.get(2)?.proof as string[],
          publicInputs: subproofs.get(2)?.publicInputs as string[],
          vkey: subproofs.get(2)?.vkey as string[],
          keyHash: subproofs.get(2)?.vkeyHash as string,
          treeHashPath: integrityCheckTreeHashPath,
          treeIndex: integrityCheckTreeIndex.toString(),
        },
        [
          {
            proof: subproofs.get(3)?.proof as string[],
            publicInputs: subproofs.get(3)?.publicInputs as string[],
            vkey: subproofs.get(3)?.vkey as string[],
            keyHash: subproofs.get(3)?.vkeyHash as string,
            treeHashPath: discloseTreeHashPath,
            treeIndex: discloseTreeIndex.toString(),
          },
          {
            proof: nationalityProof.proof as string[],
            publicInputs: nationalityProof.publicInputs as string[],
            vkey: nationalityVkey,
            keyHash: nationalityVkeyHash,
            treeHashPath: nationalityTreeHashPath,
            treeIndex: nationalityTreeIndex.toString(),
          },
          {
            proof: ageProof.proof as string[],
            publicInputs: ageProof.publicInputs as string[],
            vkey: ageVkey,
            keyHash: ageVkeyHash,
            treeHashPath: ageTreeHashPath,
            treeIndex: ageTreeIndex.toString(),
          },
        ],
        circuitManifest.root,
      )

      const proof = await outerProofCircuit.prove(inputs, {
        useCli: true,
        circuitName: "outer_count_6",
        recursive: true,
      })
      expect(proof).toBeDefined()
      const currentDate = getCurrentDateFromOuterProof(proof)
      expect(currentDate.getTime()).toEqual(nowTimestamp * 1000)
      const nullifier = getNullifierFromOuterProof(proof)
      expect(nullifier).toEqual(
        2650684516642119190462868389024749567829027632273482260700375874186000116367n,
      )
      const certificateRegistryRootFromProof = getCertificateRegistryRootFromOuterProof(proof)
      expect(certificateRegistryRoot).toEqual(certificateRegistryRootFromProof)
      const paramCommitmentsFromProof = getParamCommitmentsFromOuterProof(proof)
      expect(subproofs.get(3)?.paramCommitment).toEqual(paramCommitmentsFromProof[0])
      expect(nationalityParamCommitment).toEqual(paramCommitmentsFromProof[1])
      expect(ageParamCommitment).toEqual(paramCommitmentsFromProof[2])
      await outerProofCircuit.destroy()
    },
    60000 * 3,
  )

})
