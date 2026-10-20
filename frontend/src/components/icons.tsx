// One stroke icon family (24px grid, 2px stroke, round caps) for the whole UI.
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, className = "w-4 h-4", ...rest }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
      {...rest}
    >
      {children}
    </svg>
  );
}

export const CheckIcon = (p: IconProps) => <Icon {...p}><path d="M5 13l4 4L19 7" /></Icon>;
export const XIcon = (p: IconProps) => <Icon {...p}><path d="M6 18L18 6M6 6l12 12" /></Icon>;
export const ArrowRightIcon = (p: IconProps) => <Icon {...p}><path d="M5 12h14M13 6l6 6-6 6" /></Icon>;
export const ArrowLeftIcon = (p: IconProps) => <Icon {...p}><path d="M19 12H5M11 18l-6-6 6-6" /></Icon>;
export const ChevronDownIcon = (p: IconProps) => <Icon {...p}><path d="M6 9l6 6 6-6" /></Icon>;
export const ChevronUpIcon = (p: IconProps) => <Icon {...p}><path d="M18 15l-6-6-6 6" /></Icon>;
export const UploadIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 16V4M7 9l5-5 5 5M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" /></Icon>
);
export const WaveIcon = (p: IconProps) => <Icon {...p}><path d="M3 12h2l2-7 4 14 3-10 2 6h5" /></Icon>;
export const AlertIcon = (p: IconProps) => (
  <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 8v4M12 16h.01" /></Icon>
);
export const ThumbUpIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M7 10v11H4a1 1 0 01-1-1v-9a1 1 0 011-1h3zM7 10l4-7a2 2 0 012 2v4h5.5a2 2 0 012 2.3l-1.2 7A2 2 0 0117.3 20H7" />
  </Icon>
);
export const ThumbDownIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M17 14V3h3a1 1 0 011 1v9a1 1 0 01-1 1h-3zM17 14l-4 7a2 2 0 01-2-2v-4H5.5a2 2 0 01-2-2.3l1.2-7A2 2 0 016.7 4H17" />
  </Icon>
);
